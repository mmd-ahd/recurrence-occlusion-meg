"""Four-way multi-occlusion linear-SVM readout used for all model recognition analyses.

For every feature set (a readout stage / model condition) and every target occlusion level, images
of that level are split with stratified k-fold cross-validation. Each training split also contains
all images from the other occlusion levels; only target-level images are tested. A standardised
``LinearSVC`` classifies the four object categories jointly (chance = 25%).
"""
import os
import shutil
import tempfile
import warnings

import numpy as np
import pandas as pd
import tqdm
from joblib import Parallel, delayed, dump as jdump, load as jload
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits


def stack_features(feature_chunks):
    """Concatenate per-batch feature arrays of each condition into one float64 matrix."""
    prepared = {}
    for name, chunks in feature_chunks.items():
        X = np.concatenate(chunks, axis=0).astype(np.float64, copy=False)
        prepared[name] = np.ascontiguousarray(X)
    return prepared


def _dump_features(features, work_dir):
    """Write each feature matrix to disk so workers can memory-map it."""
    paths = {}
    for name, X in features.items():
        path = os.path.join(work_dir, f'stage_{name}.joblib')
        jdump(X, path)
        paths[name] = path
    return paths


def _default_labels(name):
    """Default label columns: the condition name as ``stage``."""
    return {'stage': name}


def _evaluate_repeat(feat_path, labels, model_name, occ_name, idx_target, idx_other, y,
                     seed, repeat, n_splits, C, tol, max_iter):
    """One cross-validation repeat for one condition and target occlusion level."""
    X = jload(feat_path, mmap_mode='r')
    X_target = np.array(X[idx_target], dtype=np.float64, order='C')
    y_target = y[idx_target]

    rows = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with threadpool_limits(limits=1):
        for fold, (train_local, test_local) in enumerate(skf.split(X_target, y_target)):
            test_idx = idx_target[test_local]
            train_idx = np.concatenate([idx_other, idx_target[train_local]])

            X_tr = np.array(X[train_idx], dtype=np.float64, order='C')
            X_te = np.array(X[test_idx], dtype=np.float64, order='C')

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            svm = LinearSVC(C=C, dual='auto', tol=tol, max_iter=max_iter, random_state=seed)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', ConvergenceWarning)
                svm.fit(X_tr_s, y[train_idx])

            rows.append({
                'model': model_name, **labels, 'occlusion': occ_name,
                'test_acc': float(np.mean(svm.predict(X_te_s) == y[test_idx])),
                'cv_mean': np.nan,  # kept for compatibility with earlier result files
                'fold': fold, 'repeat': repeat,
            })
    return rows


def evaluate_all_stages(features, metadata, model_name, label_fn=_default_labels, n_splits=8,
                        n_repeats=20, random_state=715, C=0.1, tol=1e-3, max_iter=2000,
                        n_jobs=-1, tmp_dir=None, show_progress=True):
    """Run the four-way readout on every condition in ``features``.

    Args:
        features: ``{condition_name: (n_images, n_features) array}`` from ``stack_features``.
        metadata: DataFrame with ``class``, ``occlusion`` and ``occlusion_name`` columns, one row
            per image in the same order as the feature rows.
        model_name: Value of the ``model`` column in the output.
        label_fn: Maps a condition name to extra label columns, e.g. ``{'stage': name}``.
        n_splits, n_repeats, random_state: Cross-validation folds, repeats and base seed
            (repeat ``r`` uses seed ``random_state + r``).
        C, tol, max_iter: ``LinearSVC`` settings.

    Returns:
        One row per condition, occlusion level, repeat and fold, with the test accuracy.
    """
    y = metadata['class'].to_numpy()
    occ = metadata['occlusion'].to_numpy()
    occ_mapping = metadata.groupby('occlusion')['occlusion_name'].first().to_dict()

    work_dir = tempfile.mkdtemp(prefix='svm_stage_feats_', dir=tmp_dir)
    try:
        feat_paths = _dump_features(features, work_dir)
        tasks = []
        for name in feat_paths:
            labels = label_fn(name)
            for occ_idx in np.unique(occ):
                idx_target = np.flatnonzero(occ == occ_idx)
                idx_other = np.flatnonzero(occ != occ_idx)
                for repeat in range(n_repeats):
                    tasks.append(delayed(_evaluate_repeat)(
                        feat_paths[name], labels, model_name, occ_mapping[occ_idx],
                        idx_target, idx_other, y, random_state + repeat, repeat,
                        n_splits, C, tol, max_iter))

        stream = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(tasks)
        if show_progress:
            stream = tqdm.tqdm(stream, total=len(tasks), desc=f'4-way SVM ({model_name})')
        results = [row for batch in stream for row in batch]
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return pd.DataFrame(results)


def save_results(df, results_dir):
    """Write the per-fold results and the mean-accuracy summary table."""
    os.makedirs(results_dir, exist_ok=True)
    df.to_csv(os.path.join(results_dir, 'multistage_4way_multi_occ_train_results.csv'), index=False)
    keys = [c for c in ('model', 'mechanism', 'stage_idx', 'stage') if c in df.columns]
    summary = df.groupby(keys + ['occlusion'])['test_acc'].mean().unstack()
    summary.to_csv(os.path.join(results_dir, 'summary_4way_multi_occ_train.csv'))
    return summary
