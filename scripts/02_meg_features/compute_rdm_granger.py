"""Representational Granger causality between ROIs (Fig. 3, Supp. Figs. 2-4).

For each ordered ROI pair, the target ROI's RDM at time t is predicted from the RDMs of the target
alone (reduced model) and of target plus source (full model) over a window 120-20 ms in the past,
using non-negative least squares on z-scored RDMs. The GC value is log(reduced error / full error);
the pre-stimulus mean (-50-0 ms) is stored as a baseline. Results are averaged over the 100 RDM
shuffles of ``extract_meg_rdms.py``.
Output: ``Megocclusion/derivatives/RDM_GC/sub-XX/sub-XX_occlusion-<cond>_gc_<source>-to-<target>.npz``
with ``gc_raw``, ``baseline`` and ``times`` (0-300 ms).

Usage:
    python scripts/02_meg_features/compute_rdm_granger.py [--subjects 1 2 ...] [--mask-conditions mask]
"""
import argparse
import concurrent.futures
import logging
import os

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from htrn import setup_logging
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS

log = logging.getLogger(__name__)

N_SHUFFLES = 100
ROI_PAIRS = [('V1-3', 'LOC'), ('LOC', 'IT-PHC'), ('V1-3', 'IT-PHC')]

TMIN_PREDICT, TMAX_PREDICT = 0.0, 0.300      # time range with GC estimates (s)
TMIN_BASE, TMAX_BASE = -0.050, 0.0           # baseline range (s)
PAST_START, PAST_END = -0.120, -0.020        # predictor window relative to t (s)


def _log_error_ratio(rdm_target_z, rdm_source_z, t_idx, past_indices, model):
    """log(reduced error / full error) for predicting the target RDM at ``t_idx``, or None."""
    Y = rdm_target_z[:, t_idx]
    X_target = rdm_target_z[:, past_indices]
    X_source = rdm_source_z[:, past_indices]

    model.fit(X_target, Y)
    err_reduced = np.sum((Y - model.predict(X_target)) ** 2)

    X_full = np.hstack((X_target, X_source))
    model.fit(X_full, Y)
    err_full = np.sum((Y - model.predict(X_full)) ** 2)

    if err_full > 0 and err_reduced > 0:
        return np.log(err_reduced / err_full)
    return None


def calculate_gc_trace(rdm_target_z, rdm_source_z, times, t_predict_indices, t_base_indices,
                       past_start_idx, past_end_idx):
    """GC time course over ``t_predict_indices`` and its mean over the baseline indices."""
    model = LinearRegression(positive=True)
    n_times = len(times)
    gc_raw = np.zeros(len(t_predict_indices))

    for i, t_idx in enumerate(t_predict_indices):
        past = np.arange(t_idx + past_start_idx, t_idx + past_end_idx)
        if np.any(past < 0) or np.any(past >= n_times):
            continue
        value = _log_error_ratio(rdm_target_z, rdm_source_z, t_idx, past, model)
        if value is not None:
            gc_raw[i] = value

    gc_base = []
    for t_idx in t_base_indices:
        past = np.arange(t_idx + past_start_idx, t_idx + past_end_idx)
        if np.any(past < 0):
            continue
        value = _log_error_ratio(rdm_target_z, rdm_source_z, t_idx, past, model)
        if value is not None:
            gc_base.append(value)

    return gc_raw, (np.mean(gc_base) if gc_base else 0)


def process_shuffle(sub_rdm_dir, sub_bids_id, cond_str, roi_1, roi_2, shuffle_n, times,
                    t_predict_indices, t_base_indices, past_start_idx, past_end_idx):
    """GC in both directions for one RDM shuffle, or None if its RDM files are missing."""
    def rdm_path(roi):
        return os.path.join(
            sub_rdm_dir, f'{sub_bids_id}_occlusion-{cond_str}_roi-{roi}_shuffle-{shuffle_n:02d}_rdm.npy')

    try:
        rdm_1 = np.load(rdm_path(roi_1))
        rdm_2 = np.load(rdm_path(roi_2))
    except FileNotFoundError:
        return None

    scaler = StandardScaler()
    rdm_1_z = scaler.fit_transform(rdm_1.T).T
    rdm_2_z = scaler.fit_transform(rdm_2.T).T

    args = (times, t_predict_indices, t_base_indices, past_start_idx, past_end_idx)
    gc_1_to_2, base_1_to_2 = calculate_gc_trace(rdm_2_z, rdm_1_z, *args)
    gc_2_to_1, base_2_to_1 = calculate_gc_trace(rdm_1_z, rdm_2_z, *args)
    return gc_1_to_2, base_1_to_2, gc_2_to_1, base_2_to_1


def _save_direction(sub_gc_dir, sub_bids_id, cond_str, src, dst, gcs, bases, times_predict):
    """Save the shuffle-averaged GC of one direction (skipped if no shuffle succeeded)."""
    if not gcs:
        log.warning('No successful shuffles for %s %s-to-%s', cond_str, src, dst)
        return
    np.savez(os.path.join(sub_gc_dir, f'{sub_bids_id}_occlusion-{cond_str}_gc_{src}-to-{dst}.npz'),
             gc_raw=np.mean(gcs, axis=0), baseline=np.mean(bases), times=times_predict)


def main():
    """Compute GC for every subject, condition and ROI pair."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--subjects', type=int, nargs='+', default=SUBJECT_IDS)
    parser.add_argument('--occlusion-levels', nargs='+', default=['0', '60'])
    parser.add_argument('--mask-conditions', nargs='+', default=['nomask', 'mask'])
    args = parser.parse_args()

    setup_logging()
    rdm_dir = os.path.join(DERIVATIVES_DIR, 'RDMs')
    gc_dir = os.path.join(DERIVATIVES_DIR, 'RDM_GC')
    os.makedirs(gc_dir, exist_ok=True)

    for sub_id in args.subjects:
        sub_bids_id = f'sub-{sub_id:02d}'
        log.info('Processing %s', sub_bids_id)
        sub_rdm_dir = os.path.join(rdm_dir, sub_bids_id)
        sub_gc_dir = os.path.join(gc_dir, sub_bids_id)
        os.makedirs(sub_gc_dir, exist_ok=True)

        for mask_cond in args.mask_conditions:
            for level in args.occlusion_levels:
                cond_str = level if mask_cond == 'nomask' else f'{mask_cond}-{level}'

                for roi_1, roi_2 in ROI_PAIRS:
                    times_fname = os.path.join(
                        sub_rdm_dir, f'{sub_bids_id}_occlusion-{cond_str}_roi-{roi_1}_shuffle-00_times.npy')
                    if not os.path.exists(times_fname):
                        log.warning('Missing times file for %s %s, skipping pair', cond_str, roi_1)
                        continue
                    times = np.load(times_fname)

                    t_predict = (times >= TMIN_PREDICT) & (times <= TMAX_PREDICT)
                    t_predict_indices = np.where(t_predict)[0]
                    t_base_indices = np.where((times >= TMIN_BASE) & (times < TMAX_BASE))[0]

                    sfreq = 1.0 / (times[1] - times[0])
                    past_start_idx = int(PAST_START * sfreq)
                    past_end_idx = int(PAST_END * sfreq)

                    gcs = {'12': [], '21': []}
                    bases = {'12': [], '21': []}
                    max_workers = min(os.cpu_count() or 1, N_SHUFFLES)
                    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(process_shuffle, sub_rdm_dir, sub_bids_id, cond_str, roi_1,
                                            roi_2, n, times, t_predict_indices, t_base_indices,
                                            past_start_idx, past_end_idx)
                            for n in range(N_SHUFFLES)
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                result = future.result()
                            except Exception as exc:
                                log.error('Shuffle worker error: %s', exc)
                                continue
                            if result is None:
                                continue
                            gc_12, base_12, gc_21, base_21 = result
                            gcs['12'].append(gc_12)
                            bases['12'].append(base_12)
                            gcs['21'].append(gc_21)
                            bases['21'].append(base_21)

                    times_predict = times[t_predict]
                    _save_direction(sub_gc_dir, sub_bids_id, cond_str, roi_1, roi_2,
                                    gcs['12'], bases['12'], times_predict)
                    _save_direction(sub_gc_dir, sub_bids_id, cond_str, roi_2, roi_1,
                                    gcs['21'], bases['21'], times_predict)


if __name__ == '__main__':
    main()
