"""Four-way category readout of each model's stages / conditions under occlusion (Fig. 4C, Supp. Fig. 5).

Globally pooled top-area features of every stimulus (4 categories x 3 occlusion levels x 64) are
classified with the multi-occlusion linear SVM in ``htrn.svm`` (8-fold CV, 20 repeats, C=0.1).

Models:
    htrn    readout stages LR_0..LR_6 (local recurrence) and TD_0..TD_7 (plus top-down feedback)
    blt     BLT-VS Feedforward / Recurrent / TopDownFeedback
    cornet  CORNet-Z, -RT and -S Feedforward / Recurrent

Results are written to ``results_<model folder>_stages_multi_occ_train/``.

Usage:
    python scripts/05_model_evaluation/run_stage_sweep.py --model htrn
"""
import argparse
import logging

import pandas as pd
import torch
import torch.nn.functional as F
import tqdm

from htrn import setup_logging
from htrn.config import DATA_ROOT, HTRN_CHECKPOINT, STIMULI_DIR
from htrn.data import batch_metadata, imagenet_transform, make_dataloader
from htrn.gpu import find_max_batch_size
from htrn.svm import evaluate_all_stages, save_results, stack_features

log = logging.getLogger(__name__)

LR_MAX_STAGE = 6
TD_MAX_STAGE = 7


def _htrn_label(name):
    """Split a condition name such as 'LR_6' into mechanism and stage columns."""
    mechanism, stage_idx = name.rsplit('_', 1)
    return {'mechanism': mechanism, 'stage_idx': int(stage_idx)}


def _pool(tensor):
    """Globally average-pooled activations as a NumPy array."""
    return F.adaptive_avg_pool2d(tensor, 1).flatten(1).detach().cpu().numpy()


def htrn_runs(device, args):
    """Yield ``(model_name, extract_fn, transform, label_fn, batch_size)`` for the HTRN sweep."""
    from htrn.models import generate_stage_schedule, layer_blocks, load_htrn, pooled_stage_sweep

    model = load_htrn(args.checkpoint, device=device)
    layers = layer_blocks(model)
    schedule = generate_stage_schedule()

    def extract(images):
        return pooled_stage_sweep(model, layers, schedule, images, LR_MAX_STAGE, TD_MAX_STAGE)

    yield 'DFM-ResNet', extract, imagenet_transform(), _htrn_label, args.batch_size


def blt_runs(device, args):
    """Run for BLT-VS; the batch size comes from a GPU memory probe."""
    from htrn.blt import (BLT_CONDITIONS, BLT_IT_EQUIVALENT_AREA, build_all_blt_models,
                          extract_blt_areas, get_blt_vs_transform)

    models = build_all_blt_models(device)

    def extract_one(cond, images):
        area = BLT_IT_EQUIVALENT_AREA
        act = extract_blt_areas(models[cond['name']], images, cond['timestep'], [area])[area]
        return _pool(act)

    def extract(images):
        return {c['name']: extract_one(c, images) for c in BLT_CONDITIONS}

    probe = lambda x: extract_one(BLT_CONDITIONS[-1], x)  # most expensive condition
    batch_size = find_max_batch_size(probe, device, candidates=(64, 48, 32, 16, 8, 4, 2, 1))
    yield 'BLT-VS', extract, get_blt_vs_transform(), None, batch_size


def cornet_runs(device, args):
    """Runs for each CORnet variant."""
    from htrn.cornet_variants import (CORNET_CONDITIONS, extract_areas, load_cornet_variant,
                                      pool_area)

    for variant, conditions in CORNET_CONDITIONS.items():
        model = load_cornet_variant(variant, device)

        def extract(images, variant=variant, model=model, conditions=conditions):
            return {c['name']: pool_area(extract_areas(variant, model, images, c)['IT'])
                    for c in conditions}

        yield variant, extract, imagenet_transform(), None, args.batch_size


MODELS = {
    'htrn': (htrn_runs, 'results_resnet_dfm_stages_multi_occ_train'),
    'blt': (blt_runs, 'results_blt_stages_multi_occ_train'),
    'cornet': (cornet_runs, 'results_cornet_stages_multi_occ_train'),
}


def main():
    """Extract features, run the SVM readout and save the results."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--checkpoint', default=str(HTRN_CHECKPOINT), help='HTRN checkpoint (htrn only)')
    parser.add_argument('--stimuli-dir', default=str(STIMULI_DIR))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-repeats', type=int, default=20)
    parser.add_argument('--results-dir', default=None)
    args = parser.parse_args()

    setup_logging()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    make_runs, folder = MODELS[args.model]
    results_dir = args.results_dir or str(DATA_ROOT / folder)

    all_results = []
    for model_name, extract, transform, label_fn, batch_size in make_runs(device, args):
        dataloader, _ = make_dataloader(args.stimuli_dir, transform, batch_size, args.num_workers)

        log.info('Extracting features: %s', model_name)
        chunks, metadata = {}, []
        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc=model_name):
                feats = extract(batch['image'].to(device))
                for name, feat in feats.items():
                    chunks.setdefault(name, []).append(feat)
                metadata.append(batch_metadata(batch))
        metadata = pd.concat(metadata, ignore_index=True)

        kwargs = {'label_fn': label_fn} if label_fn else {}
        all_results.append(evaluate_all_stages(
            stack_features(chunks), metadata, model_name, n_repeats=args.n_repeats, **kwargs))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = save_results(pd.concat(all_results, ignore_index=True), results_dir)
    log.info('Mean 4-way accuracy:\n%s', summary)


if __name__ == '__main__':
    main()
