"""Shared pieces of the pairwise category decoding analyses (Figs. 1-2)."""
import itertools

import mne
import numpy as np

from htrn.config import CATEGORIES, INVERSE_LAMBDA2, INVERSE_METHOD
from htrn.meg import epochs_path

CATEGORY_PAIRS = list(itertools.combinations(CATEGORIES, 2))
N_PSEUDOGROUPS = 8
N_SHUFFLES = 100
SCORING = 'roc_auc'
ANALYSIS_TMIN, ANALYSIS_TMAX = -0.1, 0.6
SVM_C = 0.1
SEED = 715


def load_pair_data(sub_bids, condition, cat_1, cat_2, inv_op, label):
    """Single-trial ROI source data of two categories.

    Args:
        condition: ``'<mask>-<level>'``, e.g. ``'nomask-60'``.

    Returns:
        ``(X_cat1, X_cat2, times)`` with trials x vertices x time, baseline-corrected, cropped to
        the analysis window and resampled to 250 Hz.
    """
    epochs = [mne.read_epochs(epochs_path(sub_bids, f'{condition}-{cat}'), verbose=False)
              for cat in (cat_1, cat_2)]
    n_ep1 = len(epochs[0])
    stcs = mne.minimum_norm.apply_inverse_epochs(
        mne.concatenate_epochs(epochs, verbose=False), inv_op, INVERSE_LAMBDA2,
        method=INVERSE_METHOD, label=label, pick_ori='normal', return_generator=False, verbose=False)

    data = []
    for stc in stcs:
        stc.apply_baseline(baseline=(None, 0), verbose=False)
        stc.crop(tmin=ANALYSIS_TMIN, tmax=ANALYSIS_TMAX)
        stc.resample(sfreq=250)
        data.append(stc.data)

    X = np.array(data)
    return X[:n_ep1], X[n_ep1:], stcs[0].times


def _pseudo_trials(X, n_groups):
    """Average randomly assigned trials into ``n_groups`` pseudo-trials (uses the global RNG)."""
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return np.array([X[g].mean(axis=0) for g in np.array_split(idx, n_groups)])


def make_pseudo_trials(X_cat1, X_cat2, n_groups=N_PSEUDOGROUPS):
    """Pseudo-trials and labels (0 = first category, 1 = second) for one shuffle."""
    X = np.concatenate([_pseudo_trials(X_cat1, n_groups), _pseudo_trials(X_cat2, n_groups)])
    y = np.concatenate([np.zeros(n_groups), np.ones(n_groups)])
    return X, y
