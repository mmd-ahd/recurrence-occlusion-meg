"""Single-trial MEG source patterns per ROI, for RSA against the models.

For each subject, mask condition and occlusion level, the epochs of all categories are resampled to
250 Hz, truncated to the trials listed in the trial-order file, projected to source space and
restricted to the vertices of each merged ROI (left and right hemispheres concatenated).
Output: ``Megocclusion/derivatives/Patterns_MEG/sub-XX/sub-XX_occlusion[-mask]-<level>_roi-<ROI>_AllCat_data.npy``
(trials x vertices x time) plus a ``*_times.npy`` file per subject.

Usage:
    python scripts/02_meg_features/extract_meg_patterns.py
"""
import logging
import os

import mne
import numpy as np
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from htrn import setup_logging
from htrn.config import CATEGORIES, DERIVATIVES_DIR, INVERSE_LAMBDA2, INVERSE_METHOD, ROIS_TO_MERGE, SUBJECT_IDS
from htrn.meg import epochs_path, merged_roi_labels
from htrn.roi_patterns import load_sequence_data

log = logging.getLogger(__name__)

OCCLUSION_LEVELS = ['0', '60', '80']
MASK_CONDITIONS = ['nomask', 'mask']


def process_subject(sub_id, sequence_data, labels_hemi, output_dir):
    """Save the single-trial ROI source patterns of one subject."""
    sub_str = f'sub-{sub_id:02d}'
    log.info('Processing %s', sub_str)

    sub_out_dir = os.path.join(output_dir, sub_str)
    os.makedirs(sub_out_dir, exist_ok=True)

    try:
        inv_op = read_inverse_operator(
            os.path.join(DERIVATIVES_DIR, 'inv_op', sub_str, f'{sub_str}-inv.fif'))
    except Exception:
        log.warning('%s: missing inverse operator', sub_str)
        return

    for mask_cond in MASK_CONDITIONS:
        for level in OCCLUSION_LEVELS:
            roi_accumulators = {roi: [] for roi in ROIS_TO_MERGE}
            times = None
            valid_level = True

            for cat in CATEGORIES:
                cond_key = f'{mask_cond}-{level}-{cat}'
                if sub_str not in sequence_data or cond_key not in sequence_data[sub_str]:
                    log.warning('%s: missing trial order for %s', sub_str, cond_key)
                    valid_level = False
                    break

                n_images = len(sequence_data[sub_str][cond_key])
                fname = epochs_path(sub_str, cond_key)
                try:
                    epochs = mne.read_epochs(fname, preload=True, verbose=False)
                    epochs.resample(250, n_jobs=1)
                except Exception:
                    log.warning('%s: missing epochs %s', sub_str, fname)
                    valid_level = False
                    break

                if len(epochs) > n_images:
                    epochs = epochs[:n_images]

                stcs = apply_inverse_epochs(epochs, inv_op, INVERSE_LAMBDA2, INVERSE_METHOD,
                                            pick_ori='normal', verbose=False)
                if times is None:
                    times = stcs[0].times

                for roi_name in ROIS_TO_MERGE:
                    lh = labels_hemi.get(f'{roi_name}-lh')
                    rh = labels_hemi.get(f'{roi_name}-rh')
                    data_list = []
                    for stc in stcs:
                        d_lh = stc.in_label(lh).data if lh else np.array([])
                        d_rh = stc.in_label(rh).data if rh else np.array([])
                        if d_lh.size and d_rh.size:
                            data_list.append(np.concatenate([d_lh, d_rh], axis=0))
                    if data_list:
                        roi_accumulators[roi_name].append(np.stack(data_list, axis=0))

            if not valid_level:
                continue

            prefix = f'occlusion-{level}' if mask_cond == 'nomask' else f'occlusion-{mask_cond}-{level}'
            for roi_name, mats in roi_accumulators.items():
                if not mats:
                    continue
                fname = os.path.join(sub_out_dir, f'{sub_str}_{prefix}_roi-{roi_name}_AllCat_data.npy')
                np.save(fname, np.concatenate(mats, axis=0))
                if roi_name == 'V1-3':
                    np.save(fname.replace('_data.npy', '_times.npy'), times)


def main():
    """Extract the MEG patterns of all subjects."""
    setup_logging()
    output_dir = os.path.join(DERIVATIVES_DIR, 'Patterns_MEG')
    os.makedirs(output_dir, exist_ok=True)

    sequence_data = load_sequence_data()
    labels_hemi = merged_roi_labels(per_hemisphere=True)
    for sub in SUBJECT_IDS:
        process_subject(sub, sequence_data, labels_hemi, output_dir)


if __name__ == '__main__':
    main()
