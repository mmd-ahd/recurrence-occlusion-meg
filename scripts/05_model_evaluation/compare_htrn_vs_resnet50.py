"""Clean ImageNet-1k accuracy of the trained HTRN vs. the torchvision ResNet-50 it started from.

Both fixed classifiers are scored on the same 50,000 validation images with the same preprocessing
(Resize 256, CenterCrop 224); the paper reports top-1/top-5 accuracy, McNemar's test on the paired
per-image correctness, and a 100,000-replicate bootstrap CI of the accuracy difference.

* HTRN: ``finetune_clean_checkpoint.pt`` run with five full-depth top-down feedback
  iterations, the number used in training (``htrn.models.sequential_topdown``).
* Stock: ``torchvision.models.resnet50(weights=IMAGENET1K_V2)``. Under this shared preprocessing its
  accuracy reads below torchvision's documented 80.858%, which uses Resize(232).

The bootstrap resamples the paired outcomes exactly by drawing from the multinomial over the 2x2
correct/incorrect table. The validation shards are downloaded from the Hugging Face hub
(``--dataset``; set ``HF_TOKEN`` or log in with ``huggingface-cli`` if the hub asks for it).

Output: ``results_htrn_vs_stock_comparison/{per_image_predictions,statistical_comparison_summary}.csv``;
existing per-image predictions of matching size are reused.

Usage:
    python scripts/05_model_evaluation/compare_htrn_vs_resnet50.py
"""
import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torchvision.transforms as transforms
import tqdm
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50 as tv_resnet50

from htrn import setup_logging
from htrn.config import DATA_ROOT, HTRN_CHECKPOINT
from htrn.gpu import find_max_batch_size
from htrn.models import load_htrn, sequential_topdown

log = logging.getLogger(__name__)

N_TD_STEPS = 5              # feedback iterations used at training time
N_BOOTSTRAP = 100_000
BOOTSTRAP_SEED = 123
DATALOADER_TIMEOUT = 300    # seconds; fail instead of hanging if workers stall

HTRN_CLEAN_TOP1_REFERENCE = 78.924        # top-1 recorded inside the checkpoint (T=5)
STOCK_DOCUMENTED_TOP1_REFERENCE = 80.858  # torchvision's number, with its own Resize(232) recipe


class ImageNetValDataset(Dataset):
    """Validation images (Resize 256, CenterCrop 224, ImageNet normalisation) from a HF dataset."""

    def __init__(self, hf_dataset, image_col, label_col):
        self.ds = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item[self.image_col]
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return self.transform(img), int(item[self.label_col])


def download_val_parquet_files(repo_id, cache_dir):
    """Download the validation shards of a HF ImageNet dataset and return their local paths."""
    info = HfApi().dataset_info(repo_id)
    val_files = sorted(s.rfilename for s in info.siblings
                       if '/val-' in s.rfilename and s.rfilename.endswith('.parquet'))
    if not val_files:
        raise RuntimeError(f"No 'val' parquet files found in {repo_id}.")
    return [hf_hub_download(repo_id=repo_id, repo_type='dataset', filename=f, cache_dir=cache_dir)
            for f in val_files]


def mcnemar_test(correct_a, correct_b):
    """McNemar's test on paired binary correctness of two classifiers on the same images.

    Uses the exact binomial test when there are fewer than 25 discordant pairs and the
    continuity-corrected chi-square otherwise.
    """
    n01 = int(np.sum(correct_a & ~correct_b))   # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))   # A wrong, B correct
    n_discordant = n01 + n10
    if n_discordant == 0:
        return {'n01': n01, 'n10': n10, 'n_discordant': 0, 'statistic': 0.0, 'p_value': 1.0,
                'method': 'degenerate'}
    if n_discordant < 25:
        res = stats.binomtest(min(n01, n10), n_discordant, 0.5)
        return {'n01': n01, 'n10': n10, 'n_discordant': n_discordant, 'statistic': None,
                'p_value': res.pvalue, 'method': 'exact_binomial'}
    chi2_stat = (abs(n01 - n10) - 1) ** 2 / n_discordant
    return {'n01': n01, 'n10': n10, 'n_discordant': n_discordant, 'statistic': chi2_stat,
            'p_value': stats.chi2.sf(chi2_stat, df=1), 'method': 'continuity_corrected_chi2'}


def bootstrap_diff_ci(correct_a, correct_b, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED, ci=0.95):
    """Bootstrap CI of accuracy_a - accuracy_b for paired outcomes on the same images.

    Resampling n images with replacement is equivalent to one Multinomial(n, table / n) draw over
    the 2x2 correct/incorrect table, so the replicates are drawn directly from that distribution.
    """
    n = len(correct_a)
    n11 = int(np.sum(correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a & correct_b))
    n00 = n - n11 - n10 - n01

    rng = np.random.default_rng(seed)
    draws = rng.multinomial(n, np.array([n11, n10, n01, n00]) / n, size=n_boot)
    diffs = (draws[:, 1] - draws[:, 2]) / n  # (a-only correct - b-only correct) / n

    lo, hi = np.percentile(diffs, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return {'n_boot': n_boot, 'observed_diff': float((n10 - n01) / n), 'mean_diff': float(diffs.mean()),
            'se': float(diffs.std(ddof=1)), 'ci_lo': float(lo), 'ci_hi': float(hi)}


def score_models(hf_ds, image_col, label_col, checkpoint, device, num_workers):
    """Per-image top-1/top-5 correctness of the stock ResNet-50 and the HTRN."""
    model_td = load_htrn(checkpoint, device=device)
    model_stock = tv_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device).eval()

    def autocast():
        return torch.autocast('cuda', dtype=torch.float16, enabled=(device.type == 'cuda'))

    def probe(fn):
        def run(x):
            with torch.inference_mode(), autocast():
                fn(x)
        return run

    bs_td = find_max_batch_size(probe(lambda x: sequential_topdown(model_td, x, N_TD_STEPS)), device, default=8)
    bs_stock = find_max_batch_size(probe(model_stock), device, default=8)
    batch_size = max(1, min(bs_td, bs_stock))

    loader = DataLoader(
        ImageNetValDataset(hf_ds, image_col, label_col), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'), persistent_workers=False,
        timeout=(DATALOADER_TIMEOUT if num_workers > 0 else 0))

    n = len(hf_ds)
    labels_arr = np.empty(n, dtype=np.int64)
    correct = {key: np.empty(n, dtype=bool)
               for key in ('stock_top1', 'stock_top5', 'td_top1', 'td_top5')}

    ptr = 0
    t0 = time.time()
    for images, labels in tqdm.tqdm(loader, desc='clean pass (both models)'):
        images = images.to(device, non_blocking=True)
        labels_gpu = labels.to(device, non_blocking=True)
        bs = labels.size(0)

        with torch.inference_mode(), autocast():
            stock_logits = model_stock(images).float()
            td_logits = model_td.classification_head(sequential_topdown(model_td, images, N_TD_STEPS)).float()

        labels_arr[ptr:ptr + bs] = labels.numpy()
        for prefix, logits in (('stock', stock_logits), ('td', td_logits)):
            top5 = logits.topk(5, dim=1).indices
            correct[f'{prefix}_top1'][ptr:ptr + bs] = (top5[:, 0] == labels_gpu).cpu().numpy()
            correct[f'{prefix}_top5'][ptr:ptr + bs] = (top5 == labels_gpu.unsqueeze(1)).any(dim=1).cpu().numpy()
        ptr += bs

    log.info('Clean pass finished in %.1f min', (time.time() - t0) / 60)
    return labels_arr, correct


def main():
    """Score both models (or reuse cached scores) and write the statistics."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--checkpoint', default=str(HTRN_CHECKPOINT))
    parser.add_argument('--dataset', default='evanarlian/imagenet_1k_resized_256', help='HF dataset with ImageNet val shards')
    parser.add_argument('--cache-dir', default=None, help='Hugging Face cache directory')
    parser.add_argument('--results-dir', default=str(DATA_ROOT / 'results_htrn_vs_stock_comparison'))
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--debug-max-images', type=int, default=None, help='Use a random subset (smoke test)')
    args = parser.parse_args()

    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.results_dir, exist_ok=True)
    suffix = '_DEBUG' if args.debug_max_images else ''
    per_image_csv = os.path.join(args.results_dir, f'per_image_predictions{suffix}.csv')
    summary_csv = os.path.join(args.results_dir, f'statistical_comparison_summary{suffix}.csv')

    hf_ds = load_dataset('parquet', data_files=download_val_parquet_files(args.dataset, args.cache_dir),
                         split='train')
    if args.debug_max_images:
        hf_ds = hf_ds.shuffle(seed=715).select(range(min(args.debug_max_images, len(hf_ds))))
    n = len(hf_ds)
    image_col = 'image' if 'image' in hf_ds.column_names else hf_ds.column_names[0]
    label_col = 'label' if 'label' in hf_ds.column_names else hf_ds.column_names[-1]
    log.info('%d validation images (image column %r, label column %r)', n, image_col, label_col)

    if os.path.exists(per_image_csv) and len(pd.read_csv(per_image_csv)) == n:
        log.info('Reusing cached per-image predictions in %s', per_image_csv)
        prior = pd.read_csv(per_image_csv)
        correct = {key: prior[f'{key}_correct'].to_numpy(dtype=bool)
                   for key in ('stock_top1', 'stock_top5', 'td_top1', 'td_top5')}
    else:
        labels_arr, correct = score_models(hf_ds, image_col, label_col, args.checkpoint, device,
                                           args.num_workers)
        pd.DataFrame({'idx': np.arange(n), 'label': labels_arr,
                      **{f'{key}_correct': v for key, v in correct.items()}}).to_csv(per_image_csv, index=False)

    for name, prefix in (('Stock_ResNet50', 'stock'), ('HTRN_TD_Sequential', 'td')):
        log.info('%-20s top-1 %.3f%%  top-5 %.3f%%', name,
                 correct[f'{prefix}_top1'].mean() * 100, correct[f'{prefix}_top5'].mean() * 100)

    summary_rows = []
    for metric in ('top1', 'top5'):
        td, stock = correct[f'td_{metric}'], correct[f'stock_{metric}']

        mc = mcnemar_test(td, stock)
        log.info('[%s] McNemar (%s): TD-only correct %d, stock-only correct %d, p=%.3e',
                 metric, mc['method'], mc['n10'], mc['n01'], mc['p_value'])
        summary_rows.append({
            'test': f"mcnemar_{mc['method']}", 'metric': metric,
            'n01_stock_only_correct': mc['n01'], 'n10_td_only_correct': mc['n10'],
            'n_discordant': mc['n_discordant'], 'statistic': mc['statistic'], 'p_value': mc['p_value'],
            'ci95_lo_pct': np.nan, 'ci95_hi_pct': np.nan,
        })

        bs = bootstrap_diff_ci(td, stock)
        log.info('[%s] difference (TD - stock) %+.3f pts, 95%% CI [%+.4f, %+.4f]', metric,
                 bs['observed_diff'] * 100, bs['ci_lo'] * 100, bs['ci_hi'] * 100)
        summary_rows.append({
            'test': 'multinomial_bootstrap_ci', 'metric': metric,
            'n01_stock_only_correct': np.nan, 'n10_td_only_correct': np.nan, 'n_discordant': np.nan,
            'statistic': bs['mean_diff'] * 100, 'p_value': np.nan,
            'ci95_lo_pct': bs['ci_lo'] * 100, 'ci95_hi_pct': bs['ci_hi'] * 100,
        })

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    log.info('Saved %s', summary_csv)
    log.info('Reference top-1: HTRN recorded in checkpoint %.3f%%; torchvision-documented stock '
             '%.3f%% (Resize 232)', HTRN_CLEAN_TOP1_REFERENCE, STOCK_DOCUMENTED_TOP1_REFERENCE)


if __name__ == '__main__':
    main()
