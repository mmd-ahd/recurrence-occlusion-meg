"""Time-resolved MEG RDMs per ROI for the representational Granger-causality analysis.

Within each subject, mask condition and occlusion level, trials of each category are shuffled and
averaged in groups of 8 (pseudo-trials), projected to source space, and the ROI patterns of all
pseudo-trials are compared with correlation distance at every time point. This is repeated for
100 shuffles; left- and right-hemisphere RDMs are averaged.
Output: ``Megocclusion/derivatives/RDMs/sub-XX/sub-XX_occlusion-<cond>_roi-<ROI>_shuffle-NN_rdm.npy``
(pairs x time) plus ``*_times.npy`` for shuffle 0.

Usage:
    python scripts/02_meg_features/extract_meg_rdms.py [--subjects 1 2 ...] [--mask-conditions mask]
"""
import argparse
import logging
import os

import mne
import numpy as np
from joblib import Parallel, delayed
from mne.minimum_norm import apply_inverse
from scipy.spatial.distance import pdist

from htrn import setup_logging
from htrn.config import (CATEGORIES, DERIVATIVES_DIR, INVERSE_LAMBDA2, INVERSE_METHOD,
                         ROIS_TO_MERGE, SUBJECT_IDS)
from htrn.meg import epochs_path, merged_roi_labels

log = logging.getLogger(__name__)

N_SHUFFLES = 100
GROUP_SIZE = 8


def compute_rdm_movie(stcs, label):
    """Correlation-distance RDM at every time point from the ROI patterns of ``stcs``.

    Returns ``(rdm (pairs x time), times)``, or ``(None, None)`` if the ROI has no data.
    """
    roi_data = []
    times = stcs[0].times

    for stc in stcs:
        try:
            data = stc.in_label(label).data
        except ValueError as exc:
            log.warning('Could not extract %s: %s', label.name, exc)
            continue
        if data.shape[0] == 0:
            log.warning('No vertices found for %s', label.name)
            continue
        roi_data.append(data)

    if not roi_data:
        return None, None

    roi_data = np.array(roi_data)
    n_groups, n_vertices, n_times = roi_data.shape
    if n_groups < 2 or n_vertices == 0:
        return None, None

    rdm = np.zeros(((n_groups * (n_groups - 1)) // 2, n_times))
    for t in range(n_times):
        rdm[:, t] = pdist(roi_data[:, :, t], metric='correlation')
    return rdm, times


def process_subject(sub_id, rdm_dir, occlusion_levels, mask_conditions, labels_hemi):
    """Compute and save the shuffled RDMs of one subject."""
    rng = np.random.default_rng(seed=42 + sub_id)
    sub_bids_id = f'sub-{sub_id:02d}'
    log.info('Processing %s', sub_bids_id)

    sub_rdm_dir = os.path.join(rdm_dir, sub_bids_id)
    os.makedirs(sub_rdm_dir, exist_ok=True)

    try:
        inv_op = mne.minimum_norm.read_inverse_operator(
            os.path.join(DERIVATIVES_DIR, 'inv_op', sub_bids_id, f'{sub_bids_id}-inv.fif'))
    except FileNotFoundError:
        log.warning('%s: inverse operator not found, skipping', sub_bids_id)
        return

    for mask_cond in mask_conditions:
        for level in occlusion_levels:
            cond_str = level if mask_cond == 'nomask' else f'{mask_cond}-{level}'

            for shuffle_n in range(N_SHUFFLES):
                stc_groups = []
                for category in CATEGORIES:
                    fname = epochs_path(sub_bids_id, f'{mask_cond}-{level}-{category}')
                    try:
                        cat_epochs = mne.read_epochs(fname, preload=True, verbose=False)
                    except FileNotFoundError:
                        log.warning('File not found %s, skipping category', fname)
                        continue

                    n_epochs = len(cat_epochs)
                    indices = np.arange(n_epochs)
                    rng.shuffle(indices)

                    n_groups = n_epochs // GROUP_SIZE
                    if n_groups == 0:
                        log.warning("Not enough epochs for '%s' (%d)", category, n_epochs)
                        continue

                    for i in range(n_groups):
                        evoked = cat_epochs[indices[i * GROUP_SIZE:(i + 1) * GROUP_SIZE]].average()
                        stc = apply_inverse(evoked, inv_op, INVERSE_LAMBDA2, INVERSE_METHOD,
                                            pick_ori='normal', verbose=False)
                        stc.apply_baseline(baseline=(None, 0))
                        stc_groups.append(stc)

                if len(stc_groups) < 2:
                    log.warning('Not enough epoch groups, skipping shuffle')
                    continue

                for roi_name in ROIS_TO_MERGE:
                    rdms, times = [], None
                    for hemi in ('lh', 'rh'):
                        label = labels_hemi.get(f'{roi_name}-{hemi}')
                        if label:
                            rdm, times_h = compute_rdm_movie(stc_groups, label)
                            if rdm is not None:
                                rdms.append(rdm)
                                times = times_h
                    if not rdms:
                        continue

                    rdm_final = sum(rdms) / len(rdms)
                    save_fname = os.path.join(
                        sub_rdm_dir,
                        f'{sub_bids_id}_occlusion-{cond_str}_roi-{roi_name}_shuffle-{shuffle_n:02d}_rdm.npy')
                    np.save(save_fname, rdm_final)
                    if shuffle_n == 0 and times is not None:
                        np.save(save_fname.replace('_rdm.npy', '_times.npy'), times)

    log.info('%s done', sub_bids_id)


def main():
    """Extract the MEG RDMs of the requested subjects in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECT_IDS)
    parser.add_argument('--occlusion-levels', nargs='+', default=['0', '60'])
    parser.add_argument('--mask-conditions', nargs='+', default=['nomask', 'mask'])
    args = parser.parse_args()

    setup_logging()
    rdm_dir = os.path.join(DERIVATIVES_DIR, 'RDMs')
    os.makedirs(rdm_dir, exist_ok=True)

    labels_hemi = merged_roi_labels(per_hemisphere=True)
    Parallel(n_jobs=-1)(
        delayed(process_subject)(sub, rdm_dir, args.occlusion_levels, args.mask_conditions, labels_hemi)
        for sub in args.subjects)


if __name__ == '__main__':
    main()
