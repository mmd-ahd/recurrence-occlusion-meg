"""Sub-averaged RSA between MEG and model patterns.

For each permutation, the trials of every category are randomly split into ``N_GROUPS`` groups and
averaged; correlation-distance RDMs are built from the group averages of the MEG data (at every
time point) and of the model features, and compared with a Spearman correlation. The result is
the correlation time course averaged over ``N_PERMUTATIONS`` random splits.
"""
import glob
import logging
import os

import numpy as np
import scipy.stats
from scipy.spatial.distance import pdist
from threadpoolctl import threadpool_limits

from htrn.config import CATEGORIES

log = logging.getLogger(__name__)

N_GROUPS = 8
N_PERMUTATIONS = 100

# (model ROI name, MEG ROI name)
ROI_MAPPINGS = [('V1_3', 'V1-3'), ('LOC', 'LOC'), ('IT', 'IT-PHC')]


def is_stale(out_path, input_paths):
    """True if ``out_path`` is missing or older than any of its inputs.

    A bare "skip if the output exists" check let results computed from earlier versions of the
    patterns be reused silently; comparing modification times keeps re-runs fast but safe.
    """
    if not os.path.exists(out_path):
        return True
    out_mtime = os.path.getmtime(out_path)
    return any(os.path.getmtime(p) > out_mtime for p in input_paths)


def get_category_slices(sub_str, mask_cond, level, sequence_data):
    """Trial index range of each category for one subject/mask/level, or None if any is missing."""
    slices = {}
    current_idx = 0
    for cat in CATEGORIES:
        cond_key = f"{mask_cond}-{level}-{cat}"
        if sub_str not in sequence_data or cond_key not in sequence_data[sub_str]:
            return None
        n_trials = len(sequence_data[sub_str][cond_key])
        slices[cat] = (current_idx, current_idx + n_trials)
        current_idx += n_trials
    return slices


def compute_subaveraged_rdm(data, slices, n_groups, seed=0):
    """Correlation-distance RDM of category sub-averages.

    Args:
        data: ``(trials, features)`` model patterns or ``(trials, channels, times)`` MEG data.
        slices: Category trial ranges from ``get_category_slices``.
        n_groups: Number of sub-averages per category.
        seed: Seed of the random trial-to-group assignment.

    Returns:
        Condensed RDM, shaped ``(pairs,)`` for 2-D data or ``(pairs, times)`` for 3-D data.
    """
    rng = np.random.default_rng(seed)
    avgs = []

    for cat in CATEGORIES:
        start, end = slices[cat]
        cdata = data[start:end]
        if len(cdata) == 0:
            continue

        idx = np.arange(len(cdata))
        rng.shuffle(idx)

        if len(cdata) < n_groups:
            idx = np.resize(idx, n_groups * (len(cdata) // n_groups + 1))[:len(cdata)]

        for grp in np.array_split(idx, n_groups):
            avgs.append(np.mean(cdata[grp], axis=0))

    stack = np.stack(avgs)

    if stack.ndim == 2:
        return pdist(stack, metric='correlation')

    if stack.ndim == 3:
        n_avg_trials = stack.shape[0]
        n_pairs = (n_avg_trials * (n_avg_trials - 1)) // 2
        n_time = stack.shape[2]

        rdms = np.zeros((n_pairs, n_time))
        for t in range(n_time):
            rdms[:, t] = pdist(stack[:, :, t], metric='correlation')
        return rdms
    return None


def normalize_meg(data, times):
    """Divide each channel by its pre-stimulus standard deviation."""
    base_mask = times < 0
    if not np.any(base_mask):
        return data
    std = np.std(data[:, :, base_mask], axis=2, keepdims=True)
    std[std == 0] = 1.0
    return data / std


def spearman_timecourse(rdm_meg_t, rdm_model):
    """Spearman correlation of each MEG RDM column with the model RDM (ranks computed once)."""
    ranked_meg = scipy.stats.rankdata(rdm_meg_t, axis=0)
    ranked_model = scipy.stats.rankdata(rdm_model)

    meg_c = ranked_meg - ranked_meg.mean(axis=0, keepdims=True)
    model_c = ranked_model - ranked_model.mean()

    num = meg_c.T @ model_c
    denom = np.sqrt(np.sum(meg_c ** 2, axis=0) * np.sum(model_c ** 2))

    with np.errstate(invalid='ignore', divide='ignore'):
        corr = num / denom

    valid = np.var(rdm_meg_t, axis=0) > 1e-12
    return np.where(valid, corr, 0.0)


def process_subject(args):
    """Compute and save the RSA time courses of one subject.

    ``args`` is ``(sub_id, sequence_data, cfg)``; ``cfg`` holds the directories, conditions and
    flags built by ``scripts/06_meg_analysis/compute_rsa.py``.
    """
    sub_id, sequence_data, cfg = args
    sub_str = f"sub-{sub_id:02d}"
    n_written = n_skipped = 0

    # One process per subject already parallelises the work; keep BLAS single-threaded.
    with threadpool_limits(limits=1):
        sub_out = os.path.join(cfg['output_dir'], sub_str)
        os.makedirs(sub_out, exist_ok=True)

        sub_meg_dir = os.path.join(cfg['meg_dir'], sub_str)
        tfile = glob.glob(os.path.join(sub_meg_dir, "*_times.npy"))
        if not tfile:
            log.warning('%s: no MEG time file', sub_str)
            return
        times = np.load(tfile[0])

        for mask_cond in cfg['mask_conditions']:
            for level in cfg['occlusion_levels']:
                slices = get_category_slices(sub_str, mask_cond, level, sequence_data)
                if not slices:
                    continue

                prefix = f"occlusion-{level}" if mask_cond == 'nomask' else f"occlusion-{mask_cond}-{level}"

                for condition in cfg['conditions']:
                    model_name = condition['model_name']

                    for mod_roi, meg_roi in ROI_MAPPINGS:
                        meg_f = os.path.join(sub_meg_dir, f"{sub_str}_{prefix}_roi-{meg_roi}_AllCat_data.npy")
                        feat_path = os.path.join(condition['source_dir'], sub_str, condition['subdir'],
                                                 f"{sub_str}_{prefix}_AllCat_ROI-{mod_roi}_t0_features.npy")
                        if not (os.path.exists(meg_f) and os.path.exists(feat_path)):
                            continue

                        out_path = os.path.join(
                            sub_out, f"{sub_str}_Model-{model_name}_ModROI-{mod_roi}_ModT-0_"
                                     f"MegROI-{meg_roi}_{prefix}_Spearman.npy")

                        if not cfg['overwrite'] and not is_stale(
                                out_path, [meg_f, feat_path, cfg['sequence_file']]):
                            n_skipped += 1
                            continue

                        meg_data = normalize_meg(np.load(meg_f), times)
                        model_features = np.load(feat_path)

                        # Slices come from the current trial-order file, so patterns extracted
                        # from an older order would silently pair the wrong trials.
                        n_trials = slices[CATEGORIES[-1]][1]
                        if model_features.shape[0] != n_trials or meg_data.shape[0] != n_trials:
                            log.warning('%s: skipping mismatched trial counts for %s %s %s '
                                        '(order file=%d, model=%d, meg=%d)', sub_str, model_name,
                                        mod_roi, prefix, n_trials, model_features.shape[0],
                                        meg_data.shape[0])
                            continue

                        rsa_time_course = np.zeros(meg_data.shape[2])
                        for p in range(N_PERMUTATIONS):
                            rdm_model = compute_subaveraged_rdm(model_features, slices, N_GROUPS, seed=p)
                            rdm_meg_t = compute_subaveraged_rdm(meg_data, slices, N_GROUPS, seed=p)
                            rsa_time_course += spearman_timecourse(rdm_meg_t, rdm_model)

                        np.save(out_path, rsa_time_course / N_PERMUTATIONS)
                        n_written += 1

    log.info('%s done: computed=%d up-to-date=%d', sub_str, n_written, n_skipped)
