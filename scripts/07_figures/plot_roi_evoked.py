"""Source-space evoked responses in V1-3, LOC and IT-PHC for 0% vs 60% occlusion (Supp. Fig. 1).

One figure per mask condition with one panel per ROI: the grand-average ROI response for each
occlusion level (mean ± SEM, smoothed). Thick segments are clusters significantly different from
zero, the red bar marks where 0% and 60% occlusion differ (cluster-permutation tests, two-tailed,
-100 to 600 ms), and dots give the peak latency (leave-one-subject-out mean ± SD). Onset latencies
are also estimated by jackknife and compared between occlusion levels with a Wilcoxon test.

Inputs:  Megocclusion/derivatives/ROIs_activity/ (``extract_roi_evoked.py``)
Outputs: figures (PNG/SVG) and latency / significance CSVs in
         Megocclusion/derivatives/ROIs_activity_figures/

Usage:
    python scripts/07_figures/plot_roi_evoked.py
"""

import logging
import os

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mne.stats import spatio_temporal_cluster_1samp_test
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

from htrn import setup_logging
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS
from htrn.figstyle import get_jackknife_onset_latencies, get_loso_latencies, save_loso_csv

log = logging.getLogger(__name__)

roi_ts_dir = os.path.join(DERIVATIVES_DIR, 'ROIs_activity')
figures_dir_base = os.path.join(DERIVATIVES_DIR, 'ROIs_activity_figures')

sub_ids = SUBJECT_IDS
occlusion_levels = ['0', '60']
mask_conditions = ['nomask', 'mask']
tmin, tmax = -0.1, 0.6

n_permutations = 'all'
p_threshold = 0.05
n_subjects = len(sub_ids)
t_threshold_cluster = stats.t.ppf(1 - p_threshold / 2, n_subjects - 1)
smoothing_sigma = 2.0

level_colors = {
    '0': '#25DE8B',
    '60': '#002C65',
}


def load_roi_timeseries():
    """ROI evoked responses of all subjects.

    Returns:
        (data, roi_names, times) where ``data[mask_cond][level]`` is (subjects, rois, times);
        missing subject files are filled with NaN.
    """
    template_sub = f'sub-{sub_ids[0]:02d}'
    template_fname = os.path.join(roi_ts_dir, template_sub,
                                  f'{template_sub}_occlusion-0_merged-rois-evoked-ts.fif')
    if not os.path.exists(template_fname):
        raise FileNotFoundError(f"Template file not found: {template_fname}")

    template_evoked = mne.read_evokeds(template_fname, verbose=False)[0]
    roi_names = template_evoked.ch_names
    times = template_evoked.times
    shape = [n_subjects, len(roi_names), len(times)]
    data = {m: {level: np.zeros(shape) for level in occlusion_levels} for m in mask_conditions}

    for i, sub in enumerate(sub_ids):
        sub_bids_id = f'sub-{sub:02d}'
        for mask_cond in mask_conditions:
            for level in occlusion_levels:
                prefix = f'occlusion-{level}' if mask_cond == 'nomask' else f'occlusion-{mask_cond}-{level}'
                fname = os.path.join(roi_ts_dir, sub_bids_id, f'{sub_bids_id}_{prefix}_merged-rois-evoked-ts.fif')
                if os.path.exists(fname):
                    evoked_ts = mne.read_evokeds(fname, verbose=False)[0]
                    if evoked_ts.ch_names != roi_names:
                        evoked_ts.pick_channels(roi_names)
                    data[mask_cond][level][i, :, :] = evoked_ts.data
                else:
                    data[mask_cond][level][i, :, :] = np.nan
    return data, roi_names, times


def baseline_significance_masks(all_label_ts, n_rois, time_mask, times_masked):
    """Boolean (rois, times) mask of clusters significantly different from zero, per condition."""
    sig_masks = {m: {} for m in mask_conditions}
    for mask_cond in mask_conditions:
        for level in occlusion_levels:
            X = all_label_ts[mask_cond][level].transpose(0, 2, 1)
            X_masked = X[:, time_mask, :]
            if np.isnan(X_masked).any():
                X_masked = np.nan_to_num(X_masked)

            _, clusters, p_vals, _ = spatio_temporal_cluster_1samp_test(
                X_masked, n_permutations=n_permutations, threshold=t_threshold_cluster,
                tail=0, n_jobs=-1, verbose=False
            )

            mask = np.zeros((n_rois, len(times_masked)), dtype=bool)
            if clusters is not None:
                for cl, p in zip(clusters, p_vals):
                    if p < p_threshold:
                        for r in range(n_rois):
                            if r in cl[1]:
                                mask[r, cl[0][cl[1] == r]] = True
            sig_masks[mask_cond][level] = mask
    return sig_masks


def main():
    """Draw the ROI evoked-response figures and save the latency and significance CSVs."""
    setup_logging()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
    os.makedirs(figures_dir_base, exist_ok=True)

    all_label_ts, roi_names, times = load_roi_timeseries()
    time_mask = (times >= tmin) & (times <= tmax)
    times_masked = times[time_mask]
    n_rois = len(roi_names)

    sig_masks_baseline = baseline_significance_masks(all_label_ts, n_rois, time_mask, times_masked)

    for mask_cond in mask_conditions:
        log.info('Mask condition: %s', mask_cond)
        loso_peaks_records = []
        loso_onsets_records = []
        onset_wilcoxon_records = []
        sig_windows_records = []

        fig, axes = plt.subplots(n_rois, 1, figsize=(10, 5 * n_rois), sharex=True, dpi=600)
        if n_rois == 1:
            axes = [axes]

        for roi_idx, (roi_name, ax) in enumerate(zip(roi_names, axes)):
            peaks_info = {}
            max_val_plot = 0
            onset_by_level = {}

            for level in occlusion_levels:
                data = all_label_ts[mask_cond][level][:, roi_idx, :]

                # Peak latency: leave-one-subject-out on |value| from stimulus onset
                peak_mask = times >= 0.0
                peaks_loso, _ = get_loso_latencies(data, times, peak_mask, smoothing_sigma,
                                                   baseline_val=0, use_abs=True)
                peak_mean = np.mean(peaks_loso)
                peak_sd = np.std(peaks_loso, ddof=1)

                # Onset is tested within 0 to tmax, the window of the baseline test
                onset_test_mask = peak_mask & (times <= tmax)
                onsets_loso, onset_mean, onset_sd = get_jackknife_onset_latencies(
                    data, times, onset_test_mask, alpha=p_threshold,
                    n_permutations=n_permutations, tail=0)
                onset_by_level[level] = {'onsets': onsets_loso, 'mean': onset_mean, 'sd': onset_sd}

                for i_sub in range(n_subjects):
                    loso_peaks_records.append({'Subject': sub_ids[i_sub], 'ROI': roi_name, 'Level': level,
                                               'Value': peaks_loso[i_sub] * 1000})
                    loso_onsets_records.append({'Subject': sub_ids[i_sub], 'ROI': roi_name, 'Level': level,
                                                'Value': onsets_loso[i_sub] * 1000})

                mean_s = gaussian_filter1d(np.mean(data, axis=0), sigma=smoothing_sigma)
                err_s = gaussian_filter1d(sem(data, axis=0, nan_policy='omit'), sigma=smoothing_sigma)

                max_val_plot = max(max_val_plot, np.max(mean_s + err_s))
                col = level_colors[level]

                ax.plot(times, mean_s, color=col, linewidth=2.5)
                ax.fill_between(times, mean_s - err_s, mean_s + err_s, color=col, alpha=0.1, edgecolor=col)

                if np.any(sig_masks_baseline[mask_cond][level][roi_idx]):
                    full_mask = np.zeros(len(times), dtype=bool)
                    full_mask[time_mask] = sig_masks_baseline[mask_cond][level][roi_idx]

                    ax.plot(times, np.where(full_mask, mean_s, np.nan), color=col, linewidth=4.5)

                    peak_time = peak_mean
                    time_idx = np.argmin(np.abs(times - peak_time))
                    peak_val = mean_s[time_idx]

                    peaks_info[level] = {'time': peak_time, 'sd': peak_sd, 'val': peak_val, 'color': col}
                    ax.plot(peak_time, peak_val, 'o', color=col, markersize=5, zorder=10,
                            markeredgecolor='white')

            # Onset latency: 0% vs 60% occlusion
            if '0' in onset_by_level and '60' in onset_by_level:
                onsets_0 = onset_by_level['0']['onsets']
                onsets_60 = onset_by_level['60']['onsets']
                if not np.any(np.isnan(onsets_0)) and not np.any(np.isnan(onsets_60)):
                    onset_stat, onset_p = stats.wilcoxon(onsets_60 * 1000, onsets_0 * 1000)
                else:
                    onset_stat, onset_p = np.nan, np.nan
                onset_wilcoxon_records.append({
                    'ROI': roi_name,
                    'Onset_Mean_0': onset_by_level['0']['mean'] * 1000,
                    'Onset_SD_0': onset_by_level['0']['sd'] * 1000,
                    'Onset_Mean_60': onset_by_level['60']['mean'] * 1000,
                    'Onset_SD_60': onset_by_level['60']['sd'] * 1000,
                    'Wilcoxon_Statistic': onset_stat,
                    'p_value': onset_p,
                })

            # Windows where 0% and 60% occlusion differ
            data_diff = (all_label_ts[mask_cond]['0'][:, roi_idx, time_mask]
                         - all_label_ts[mask_cond]['60'][:, roi_idx, time_mask])
            _, clusters_diff, cluster_p_diff, _ = spatio_temporal_cluster_1samp_test(
                data_diff[:, :, np.newaxis], threshold=t_threshold_cluster,
                n_permutations=n_permutations, tail=0, n_jobs=-1, verbose=False,
                out_type='indices'
            )

            sig_clusters_diff = [c for c, p in zip(clusters_diff, cluster_p_diff) if p < p_threshold]
            if sig_clusters_diff:
                trans = ax.get_xaxis_transform()
                for c, p in zip(clusters_diff, cluster_p_diff):
                    if p >= p_threshold:
                        continue
                    time_indices = c[0]
                    ax.plot(times_masked[time_indices], [0.03] * len(time_indices), transform=trans,
                            color='#ff0004', linewidth=4.5)
                    sig_windows_records.append({
                        'ROI': roi_name,
                        'Comparison': f'{mask_cond} 0% vs 60%',
                        'Start_Time_ms': times_masked[time_indices[0]] * 1000,
                        'End_Time_ms': times_masked[time_indices[-1]] * 1000,
                        'P_Value': p
                    })

            # Stagger the peak labels so they do not overlap
            if len(peaks_info) == 2:
                t0 = peaks_info['0']['time']
                t60 = peaks_info['60']['time']

                if abs(t0 - t60) < 0.05:
                    if peaks_info['0']['val'] > peaks_info['60']['val']:
                        top_level, bot_level = '0', '60'
                    else:
                        top_level, bot_level = '60', '0'
                    peaks_info[top_level]['offset_mult'] = 0.16
                    peaks_info[bot_level]['offset_mult'] = 0.14
                else:
                    peaks_info['0']['offset_mult'] = 0.14
                    peaks_info['60']['offset_mult'] = 0.16
            elif len(peaks_info) == 1:
                peaks_info[list(peaks_info)[0]]['offset_mult'] = 0.12

            for level, info in peaks_info.items():
                ax.text(info['time'], info['val'] + max_val_plot * info['offset_mult'],
                        f"{info['time'] * 1000:.0f} ± {info['sd'] * 1000:.0f} ms",
                        ha='center', va='bottom', fontsize=16, fontweight='bold', color=info['color'])

            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
            ax.axvline(0, color='gray', linestyle=':', linewidth=1.5)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel('Activation (AU)', fontsize=22)

            title_prefix = "No Mask - " if mask_cond == 'nomask' else "Mask - "
            ax.set_title(f'{title_prefix}{roi_name}', fontsize=24, fontweight='bold', loc='center')
            ax.tick_params(labelsize=20)

            if roi_idx == 0:
                legend_handles = [
                    Line2D([0], [0], color='#4d4d4d', lw=4.5),
                    Line2D([0], [0], color='#4d4d4d', lw=2),
                    (Patch(facecolor='#4d4d4d', alpha=0.4, edgecolor="#4d4d4d"),
                     Line2D([0], [0], color='#4d4d4d', lw=2)),
                ]
                legend_labels = ['S.', 'N. S.', 'SEM']  # significant, not significant
                for level in occlusion_levels:
                    legend_handles.append(Line2D([0], [0], color=level_colors[level], lw=6))
                    legend_labels.append(f'{level}%')

                ax.legend(handles=legend_handles, labels=legend_labels,
                          fontsize=20, title_fontsize=22, frameon=False, loc='upper right', ncol=2)

        axes[-1].set_xlabel('Time (s)', fontsize=22)
        axes[-1].set_xlim(tmin, tmax)

        save_loso_csv(loso_peaks_records, os.path.join(figures_dir_base, f'ROIs_Activity_LOSO_Peak_Latencies_{mask_cond}.csv'))
        save_loso_csv(loso_onsets_records, os.path.join(figures_dir_base, f'ROIs_Activity_LOSO_Onset_Latencies_{mask_cond}.csv'))
        if onset_wilcoxon_records:
            pd.DataFrame(onset_wilcoxon_records).to_csv(
                os.path.join(figures_dir_base, f'ROIs_Activity_Onset_Wilcoxon_0_vs_60_{mask_cond}.csv'), index=False)
        if sig_windows_records:
            pd.DataFrame(sig_windows_records).to_csv(
                os.path.join(figures_dir_base, f'ROIs_Activity_Significance_Windows_{mask_cond}.csv'), index=False)

        plt.tight_layout()
        fig.savefig(os.path.join(figures_dir_base, f'ROIs_Activity_0_vs_60_mean_{mask_cond}.png'))
        fig.savefig(os.path.join(figures_dir_base, f'ROIs_Activity_0_vs_60_mean_{mask_cond}.svg'))
        plt.close(fig)


if __name__ == '__main__':
    main()
