"""Time-resolved pairwise category decoding in source-space ROIs (Fig. 1).

For every subject, occlusion level, ROI and category pair, trials are averaged into 8 pseudo-trials
per category and a linear SVM is cross-validated (8-fold, ROC-AUC) at each time point; this is
repeated for 100 random pseudo-trial assignments and averaged. Scores are averaged over the six
category pairs.
Output: ``Megocclusion/derivatives/decoding/sub-XX/<level>/<cat1>_<cat2>/scores_<ROI>.npy`` and
``scores_<ROI>_avg-pairs.npy``.

Usage:
    python scripts/06_meg_analysis/decode_categories.py
"""
import os

# Single-threaded BLAS: parallelism comes from one process per subject
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[var] = '1'

import argparse  # noqa: E402
import logging  # noqa: E402

import mne  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from mne.decoding import SlidingEstimator, Vectorizer, cross_val_multiscore  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from htrn import setup_logging  # noqa: E402
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS  # noqa: E402
from htrn.decoding import (CATEGORY_PAIRS, N_PSEUDOGROUPS, N_SHUFFLES, SCORING, SEED, SVM_C,  # noqa: E402
                           load_pair_data, make_pseudo_trials)
from htrn.meg import merged_roi_labels  # noqa: E402

log = logging.getLogger(__name__)

CV_FOLDS = 8


def process_subject(sub_id, decoding_dir, mask_conditions, occlusion_levels, labels, first_subject):
    """Decode all category pairs, ROIs and conditions for one subject."""
    setup_logging()
    sub_bids = f'sub-{sub_id:02d}'
    log.info('Subject %s', sub_bids)
    sub_decoding_dir = os.path.join(decoding_dir, sub_bids)
    os.makedirs(sub_decoding_dir, exist_ok=True)
    inv_op = mne.minimum_norm.read_inverse_operator(
        os.path.join(DERIVATIVES_DIR, 'inv_op', sub_bids, f'{sub_bids}-inv.fif'))

    for mask_cond in mask_conditions:
        for level in occlusion_levels:
            dir_cond_name = level if mask_cond == 'nomask' else f'{mask_cond}-{level}'
            sub_level_dir = os.path.join(sub_decoding_dir, dir_cond_name)
            os.makedirs(sub_level_dir, exist_ok=True)

            for roi, roi_label in labels.items():
                pair_scores = []
                for cat_1, cat_2 in CATEGORY_PAIRS:
                    X_cat1, X_cat2, times = load_pair_data(
                        sub_bids, f'{mask_cond}-{level}', cat_1, cat_2, inv_op, roi_label)

                    shuffle_scores = []
                    for shuffle_idx in range(N_SHUFFLES):
                        X, y = make_pseudo_trials(X_cat1, X_cat2, N_PSEUDOGROUPS)
                        pipeline = make_pipeline(Vectorizer(), StandardScaler(),
                                                 SVC(kernel='linear', C=SVM_C, random_state=SEED))
                        sliding = SlidingEstimator(pipeline, scoring=SCORING, n_jobs=1)
                        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED + shuffle_idx)
                        scores = cross_val_multiscore(sliding, X, y, cv=cv, n_jobs=1)
                        shuffle_scores.append(scores.mean(axis=0))

                    pair_score = np.mean(shuffle_scores, axis=0)
                    pair_scores.append(pair_score)

                    if first_subject:
                        np.save(os.path.join(decoding_dir, 'time_points.npy'), times)
                    pair_dir = os.path.join(sub_level_dir, f'{cat_1}_{cat_2}')
                    os.makedirs(pair_dir, exist_ok=True)
                    np.save(os.path.join(pair_dir, f'scores_{roi}.npy'), pair_score)

                np.save(os.path.join(sub_level_dir, f'scores_{roi}_avg-pairs.npy'),
                        np.mean(pair_scores, axis=0))

    return f'Completed {sub_bids}'


def main():
    """Run the decoding for all subjects in parallel."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECT_IDS)
    parser.add_argument('--occlusion-levels', nargs='+', default=['0', '60'])
    parser.add_argument('--mask-conditions', nargs='+', default=['nomask'])
    parser.add_argument('--n-jobs', type=int, default=14)
    args = parser.parse_args()

    setup_logging()
    decoding_dir = os.path.join(DERIVATIVES_DIR, 'decoding')
    os.makedirs(decoding_dir, exist_ok=True)

    labels = merged_roi_labels()
    Parallel(n_jobs=args.n_jobs)(
        delayed(process_subject)(sub, decoding_dir, args.mask_conditions, args.occlusion_levels,
                                 labels, sub == args.subjects[0])
        for sub in args.subjects)


if __name__ == '__main__':
    main()
