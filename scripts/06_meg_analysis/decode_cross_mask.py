"""Generalization of category decoding across backward masking (Fig. 2).

A linear SVM trained on masked trials is tested on unmasked trials of the same occlusion level, at
every time point (pairwise categories, 8 pseudo-trials per category, ROC-AUC, 100 random
pseudo-trial assignments averaged). Scores are averaged over the six category pairs.
Output: ``Megocclusion/derivatives/decoding_cross_condition/sub-XX/train-mask_test-nomask_<level>/``
with the same file layout as ``decode_categories.py``.

Usage:
    python scripts/06_meg_analysis/decode_cross_mask.py
"""
import argparse
import logging
import os

import mne
import numpy as np
from mne.decoding import SlidingEstimator, Vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from htrn import setup_logging
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS
from htrn.decoding import (CATEGORY_PAIRS, N_PSEUDOGROUPS, N_SHUFFLES, SCORING, SEED, SVM_C,
                           load_pair_data, make_pseudo_trials)
from htrn.meg import merged_roi_labels

log = logging.getLogger(__name__)


def main():
    """Run the mask-to-no-mask decoding for all subjects."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECT_IDS)
    parser.add_argument('--occlusion-levels', nargs='+', default=['0', '60'])
    args = parser.parse_args()

    setup_logging()
    decoding_dir = os.path.join(DERIVATIVES_DIR, 'decoding_cross_condition')
    os.makedirs(decoding_dir, exist_ok=True)
    labels = merged_roi_labels()

    for sub_id in args.subjects:
        sub_bids = f'sub-{sub_id:02d}'
        log.info('Subject %s', sub_bids)
        sub_decoding_dir = os.path.join(decoding_dir, sub_bids)
        os.makedirs(sub_decoding_dir, exist_ok=True)
        inv_op = mne.minimum_norm.read_inverse_operator(
            os.path.join(DERIVATIVES_DIR, 'inv_op', sub_bids, f'{sub_bids}-inv.fif'))

        for level in args.occlusion_levels:
            sub_level_dir = os.path.join(sub_decoding_dir, f'train-mask_test-nomask_{level}')
            os.makedirs(sub_level_dir, exist_ok=True)

            for roi, roi_label in labels.items():
                pair_scores = []
                for cat_1, cat_2 in CATEGORY_PAIRS:
                    X_train_1, X_train_2, times = load_pair_data(
                        sub_bids, f'mask-{level}', cat_1, cat_2, inv_op, roi_label)
                    X_test_1, X_test_2, _ = load_pair_data(
                        sub_bids, f'nomask-{level}', cat_1, cat_2, inv_op, roi_label)

                    shuffle_scores = []
                    for _ in range(N_SHUFFLES):
                        X_train, y_train = make_pseudo_trials(X_train_1, X_train_2, N_PSEUDOGROUPS)
                        X_test, y_test = make_pseudo_trials(X_test_1, X_test_2, N_PSEUDOGROUPS)

                        pipeline = make_pipeline(Vectorizer(), StandardScaler(),
                                                 SVC(kernel='linear', C=SVM_C, random_state=SEED))
                        sliding = SlidingEstimator(pipeline, scoring=SCORING, n_jobs=-1)
                        sliding.fit(X_train, y_train)
                        shuffle_scores.append(sliding.score(X_test, y_test))

                    pair_score = np.mean(shuffle_scores, axis=0)
                    pair_scores.append(pair_score)

                    np.save(os.path.join(decoding_dir, 'time_points.npy'), times)
                    pair_dir = os.path.join(sub_level_dir, f'{cat_1}_{cat_2}')
                    os.makedirs(pair_dir, exist_ok=True)
                    np.save(os.path.join(pair_dir, f'scores_{roi}.npy'), pair_score)

                np.save(os.path.join(sub_level_dir, f'scores_{roi}_avg-pairs.npy'),
                        np.mean(pair_scores, axis=0))


if __name__ == '__main__':
    main()
