"""Evoked responses in the merged ROIs (V1-3, LOC, IT-PHC) per subject, mask condition and level.

Trials of all categories are averaged, projected to source space, and the ROI time course is taken
with ``mean_flip`` and sign-aligned so that its largest deflection is positive.
Output: ``Megocclusion/derivatives/ROIs_activity/sub-XX/sub-XX_occlusion[-mask]-<level>_merged-rois-evoked-ts.fif``

Usage:
    python scripts/02_meg_features/extract_roi_evoked.py
"""
import logging
import os

import mne
import numpy as np

from htrn import setup_logging
from htrn.config import CATEGORIES, DERIVATIVES_DIR, INVERSE_LAMBDA2, INVERSE_METHOD, SUBJECT_IDS
from htrn.meg import epochs_path, merged_roi_labels

log = logging.getLogger(__name__)

OCCLUSION_LEVELS = ['0', '60', '80']
MASK_CONDITIONS = ['nomask', 'mask']


def main():
    """Compute and save the ROI evoked responses of every subject."""
    setup_logging()
    inv_op_dir = os.path.join(DERIVATIVES_DIR, 'inv_op')
    stc_dir = os.path.join(DERIVATIVES_DIR, 'ROIs_activity')
    os.makedirs(stc_dir, exist_ok=True)

    labels = list(merged_roi_labels().values())
    label_names = [label.name for label in labels]

    for sub in SUBJECT_IDS:
        sub_bids_id = f'sub-{sub:02d}'
        log.info('Processing %s', sub_bids_id)
        sub_stc_dir = os.path.join(stc_dir, sub_bids_id)
        os.makedirs(sub_stc_dir, exist_ok=True)

        try:
            inv_op = mne.minimum_norm.read_inverse_operator(
                os.path.join(inv_op_dir, sub_bids_id, f'{sub_bids_id}-inv.fif'))
        except FileNotFoundError:
            log.warning('%s: inverse operator not found, skipping', sub_bids_id)
            continue

        for mask_cond in MASK_CONDITIONS:
            for level in OCCLUSION_LEVELS:
                fnames = [epochs_path(sub_bids_id, f'{mask_cond}-{level}-{cat}') for cat in CATEGORIES]
                epochs_list = [mne.read_epochs(f, preload=True, verbose=False)
                               for f in fnames if os.path.isfile(f)]
                if not epochs_list:
                    log.warning('%s: no epochs for %s level %s', sub_bids_id, mask_cond, level)
                    continue

                epochs = mne.concatenate_epochs(epochs_list, verbose=False)
                epochs.apply_baseline(baseline=(None, 0), verbose=False)

                stc = mne.minimum_norm.apply_inverse(epochs.average(), inv_op, INVERSE_LAMBDA2,
                                                     method=INVERSE_METHOD, pick_ori=None, verbose=False)
                stc.apply_baseline(baseline=(None, 0), verbose=False)

                label_ts = mne.extract_label_time_course(
                    stc, labels, inv_op['src'], mode='mean_flip', return_generator=False, verbose=False)

                # Make the largest deflection of each ROI positive
                for i in range(len(label_ts)):
                    if label_ts[i, np.argmax(np.abs(label_ts[i, :]))] < 0:
                        label_ts[i, :] *= -1

                info = mne.create_info(ch_names=label_names, sfreq=epochs.info['sfreq'], ch_types='misc')
                roi_evoked = mne.EvokedArray(np.array(label_ts), info, tmin=epochs.tmin)

                prefix = f'occlusion-{level}' if mask_cond == 'nomask' else f'occlusion-{mask_cond}-{level}'
                roi_evoked.save(os.path.join(sub_stc_dir, f'{sub_bids_id}_{prefix}_merged-rois-evoked-ts.fif'),
                                overwrite=True, verbose=False)


if __name__ == '__main__':
    main()
