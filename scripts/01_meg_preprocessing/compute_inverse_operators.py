"""Compute one minimum-norm inverse operator per subject.

The noise covariance is estimated from the pre-stimulus baseline of all 24 conditions, with the
subject's data rank fixed after ICA. Output: ``Megocclusion/derivatives/inv_op/sub-XX/sub-XX-inv.fif``

Usage:
    python scripts/01_meg_preprocessing/compute_inverse_operators.py
"""
import gc
import logging
import os

import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

from htrn import setup_logging
from htrn.config import DATASET_DIR, DERIVATIVES_DIR
from htrn.meg import EVENT_ID_MAP, epochs_path

log = logging.getLogger(__name__)

# Rank of each subject's MEG data after ICA cleaning
SUBJECT_RANKS = {
    1: 70, 2: 72, 3: 71, 4: 73, 5: 59, 6: 70, 7: 67, 8: 69,
    9: 70, 10: 70, 12: 65, 13: 68, 14: 71, 15: 66,
}


def main():
    """Compute and save one inverse operator per subject."""
    setup_logging()
    output_dir = os.path.join(DERIVATIVES_DIR, 'inv_op')
    os.makedirs(output_dir, exist_ok=True)

    fwd = mne.read_forward_solution(os.path.join(DATASET_DIR, 'fwd', 'fsaverage-meg-oct6-fwd.fif'))
    if fwd['src'][0]['dist'] is None or fwd['src'][1]['dist'] is None:
        mne.add_source_space_distances(fwd['src'], dist_limit=0.04, n_jobs=-1)
    fwd = mne.convert_forward_solution(fwd, surf_ori=True)

    for sub, rank in SUBJECT_RANKS.items():
        sub_bids_id = f'sub-{sub:02d}'
        sub_output_dir = os.path.join(output_dir, sub_bids_id)
        os.makedirs(sub_output_dir, exist_ok=True)

        epochs_list = [
            mne.read_epochs(epochs_path(sub_bids_id, cond), preload=True, verbose=False)
            for cond in EVENT_ID_MAP if os.path.isfile(epochs_path(sub_bids_id, cond))
        ]
        if not epochs_list:
            log.warning('%s: no epoch files found, skipping', sub_bids_id)
            continue

        log.info('%s: %d conditions', sub_bids_id, len(epochs_list))
        all_epochs = mne.concatenate_epochs(epochs_list)
        del epochs_list
        gc.collect()

        all_epochs.apply_baseline(baseline=(None, 0), verbose=False)
        noise_cov = mne.compute_covariance(all_epochs, tmax=0, method='shrunk',
                                           rank={'meg': rank}, verbose=False)
        inv_op = make_inverse_operator(all_epochs.info, fwd, noise_cov, loose=0.2, depth=0.8,
                                       rank={'meg': rank}, verbose=False)
        write_inverse_operator(os.path.join(sub_output_dir, f'{sub_bids_id}-inv.fif'), inv_op,
                               overwrite=True, verbose=False)

        del all_epochs, noise_cov, inv_op
        gc.collect()


if __name__ == '__main__':
    main()
