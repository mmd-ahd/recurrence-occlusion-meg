"""Grand-average source estimate for unoccluded, unmasked trials (0% occlusion, no mask).

Per subject, the evoked response over all categories is projected to source space with the
subject's inverse operator; the estimates are then averaged across subjects on fsaverage.
Output: ``Megocclusion/derivatives/grand_average_stc/grand-average_occlusion-0-nomask_meg-{lh,rh}.stc``

Usage:
    python scripts/02_meg_features/estimate_grand_average_source.py
"""
import logging
import os

import mne
import numpy as np

from htrn import setup_logging
from htrn.config import CATEGORIES, DERIVATIVES_DIR, INVERSE_LAMBDA2, INVERSE_METHOD, SUBJECT_IDS
from htrn.meg import epochs_path

log = logging.getLogger(__name__)

LEVEL = '0'
MASK_COND = 'nomask'


def main():
    """Average the per-subject source estimates and save the result."""
    setup_logging()
    inv_op_dir = os.path.join(DERIVATIVES_DIR, 'inv_op')
    output_dir = os.path.join(DERIVATIVES_DIR, 'grand_average_stc')
    os.makedirs(output_dir, exist_ok=True)

    stcs = []
    for sub in SUBJECT_IDS:
        sub_bids_id = f'sub-{sub:02d}'
        inv_fname = os.path.join(inv_op_dir, sub_bids_id, f'{sub_bids_id}-inv.fif')
        try:
            inv_op = mne.minimum_norm.read_inverse_operator(inv_fname)
        except FileNotFoundError:
            log.warning('%s: inverse operator not found, skipping', sub_bids_id)
            continue

        fnames = [epochs_path(sub_bids_id, f'{MASK_COND}-{LEVEL}-{cat}') for cat in CATEGORIES]
        epochs_list = [mne.read_epochs(f, preload=True, verbose=False) for f in fnames if os.path.isfile(f)]
        if not epochs_list:
            log.warning('%s: no epochs found, skipping', sub_bids_id)
            continue

        epochs = mne.concatenate_epochs(epochs_list, verbose=False)
        epochs.apply_baseline(baseline=(None, 0), verbose=False)
        stc = mne.minimum_norm.apply_inverse(epochs.average(), inv_op, INVERSE_LAMBDA2,
                                             method=INVERSE_METHOD, pick_ori=None, verbose=False)
        stc.apply_baseline(baseline=(None, 0), verbose=False)
        stcs.append(stc)

    if not stcs:
        raise RuntimeError('No source estimates were computed for any subject.')

    template = stcs[0]
    grand_average = mne.SourceEstimate(
        np.mean([stc.data for stc in stcs], axis=0), vertices=template.vertices,
        tmin=template.tmin, tstep=template.tstep, subject='fsaverage')
    grand_average.save(os.path.join(output_dir, 'grand-average_occlusion-0-nomask_meg'),
                       overwrite=True, verbose=False)
    log.info('Saved grand average of %d subjects', len(stcs))


if __name__ == '__main__':
    main()
