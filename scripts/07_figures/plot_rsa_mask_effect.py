"""Effect of backward masking on HTRN model-brain RSA (Supp. Figs. 6-7).

For each HTRN variant (rows: FF, LR, TD) and ROI (columns), model-brain Spearman correlations over
time for no-mask and mask trials, one figure per occlusion level (0%, 60%). Curves show the mean
over subjects with SEM; thick segments mark clusters significantly above baseline. The rail below
each panel marks where no-mask and mask differ, in the colour of the condition with the higher
correlation (cluster-permutation test on 0-400 ms; only stretches where at least one condition is
itself above baseline are reported). Leave-one-subject-out peak latencies are listed above each panel.

Inputs:  Megocclusion/derivatives/RSA_Subaveraged_Results_ResNet/ (``compute_rsa.py --model htrn``)
Outputs: figure (PNG/SVG/PDF), latency CSVs and significance-window CSVs in
         <RSA folder>/plots_Mask_Effect_v2/

Usage:
    python scripts/07_figures/plot_rsa_mask_effect.py
"""

import glob
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.stats import sem

from htrn import figstyle as fs, setup_logging
from htrn.config import DERIVATIVES_DIR, SUBJECT_IDS
from htrn.figstyle import (
    INK, MID, HAIRLINE, GROUND, MASK_COLOR,
    get_loso_latencies, get_jackknife_onset_latencies, save_loso_csv, run_clusters,
    style_axes, draw_legend_row, draw_ledger, draw_rail_single, draw_cascade,
    save_figure, report_page_fit,
)

log = logging.getLogger(__name__)
fs.setup_rcparams()

derivatives_dir = str(DERIVATIVES_DIR)
rsa_dir = os.path.join(derivatives_dir, 'RSA_Subaveraged_Results_ResNet')
meg_pattern_dir = os.path.join(derivatives_dir, 'Patterns_MEG')
out_dir = os.path.join(rsa_dir, 'plots_Mask_Effect_v2')

sub_ids = SUBJECT_IDS
occlusion_levels = ['0', '60']
mask_conditions = ['nomask', 'mask']

# (model ROI, MEG ROI); columns of the figure
ROI_PAIRS = [
    ('V1_3', 'V1-3'),
    ('LOC',  'LOC'),
    ('IT',   'IT-PHC')
]
ROI_DISPLAY = {'V1_3': 'V1-3', 'LOC': 'LOC', 'IT': 'IT-PHC'}
# Model layers compared with each ROI, shown in the column titles
LAYER_DISPLAY = {'V1_3': 'Conv2-3', 'LOC': 'Conv4', 'IT': 'Conv5'}

# (RSA model name, label, short label, colour, linewidth, peak marker); the HTRN variants are
# readout stages 0 / 6 / 7 of the same checkpoint
MECHANISMS = [
    ('DFM-Feedforward',     'Feedforward (FF)',       'FF', '#F4256D', fs.pt(1.30), 'o'),
    ('DFM-Recurrent',       'Local-Recurrent (LR)',  'LR', '#3A0CA3', fs.pt(1.50), 's'),
    ('DFM-TopDownFeedback', 'Top-Down (TD)',         'TD', '#00A5A8', fs.pt(1.50), 'D'),
]
ROW_LABELS = ['Feedforward', 'Local-Recurrent', 'Top-Down']

MASK_LW = fs.pt(1.20)

# The axes start at -50 ms, but tests and the peak search use t >= 0 only
TMIN, TMAX = -0.05, 0.4
TICK_STEP = 0.05
ALPHA = 0.05
SIGMA_SMOOTH = 2.0
N_PERMUTATIONS = 'all'

# Layout: one row per variant, one column per ROI; the header is tall enough for a two-line legend
FIGSIZE = (fs.FIG_W, 11.95)
RAIL_RATIO = 0.16                # one comparison lane per panel
GRID = dict(hspace=0.13, wspace=0.13, left=0.098, right=0.988, top=0.884, bottom=0.048)
LEGEND_Y = 0.986
LEGEND_DY = 0.020
OCCLUSION_TAG_Y = 0.938
COLTITLE_Y = 0.900


def style_time_axis(ax, ticklabels=False, label=False):
    """Time axis with this figure's limits and tick spacing."""
    fs.style_time_axis(ax, TMIN, TMAX, TICK_STEP, ticklabels=ticklabels, label=label)


SIG_WINDOW_COLUMNS = ['Mechanism', 'ROI', 'Level', 'Comparison', 'Winner', 'Loser',
                      'Start_Time_ms', 'End_Time_ms', 'P_Value', 'Stat_Window']


def load_meg_time_vector():
    """Time vector of the MEG patterns (default -100 to 600 ms at 250 Hz if none is found)."""
    try:
        files = glob.glob(os.path.join(meg_pattern_dir, "sub-*", "*_times.npy"))
        if files:
            return np.load(files[0])
    except Exception:
        pass
    return np.linspace(-0.1, 0.6, 176)


def load_rsa_data(model_name, mod_roi, meg_roi, level, mask_cond):
    """(subjects, times) RSA correlations, or None if any subject's file is missing."""
    group_data = []
    if mask_cond == 'nomask':
        file_prefix = f"occlusion-{level}"
    else:
        file_prefix = f"occlusion-{mask_cond}-{level}"

    for sub in sub_ids:
        sub_str = f"sub-{sub:02d}"
        fname = (f"{sub_str}_Model-{model_name}_ModROI-{mod_roi}_ModT-0_"
                 f"MegROI-{meg_roi}_{file_prefix}_Spearman.npy")
        fpath = os.path.join(rsa_dir, sub_str, fname)
        if os.path.exists(fpath):
            group_data.append(np.load(fpath))

    if len(group_data) < len(sub_ids):
        return None
    return np.array(group_data)


def main():
    """Draw one figure per occlusion level and save the CSVs."""
    setup_logging()
    os.makedirs(out_dir, exist_ok=True)

    full_times = load_meg_time_vector()
    baseline_mask = full_times < 0
    plot_mask = (full_times >= TMIN) & (full_times <= TMAX)
    plot_times = full_times[plot_mask]

    n_subjects = len(sub_ids)
    t_threshold = stats.t.ppf(1 - ALPHA / 2, n_subjects - 1)
    peak_mask = plot_times >= 0.0
    stat_mask = plot_times >= 0.0
    n_rows = len(MECHANISMS)

    for level in occlusion_levels:
        log.info('Occlusion level: %s%%', level)
        loso_peaks_records = []
        loso_onsets_records = []
        sig_windows_records = []

        fig = plt.figure(figsize=FIGSIZE, dpi=110, facecolor=GROUND)
        outer = fig.add_gridspec(n_rows, len(ROI_PAIRS), **GRID)

        legend_items = [
            (label, color, marker, fs.MS_LEGEND if marker != 'D' else fs.MS_LEGEND_SM, 'solid', lw)
            for (_, label, short, color, lw, marker) in MECHANISMS
        ]
        legend_items.append(('Mask', MASK_COLOR, 'v', fs.MS_LEGEND_SM, 'solid', MASK_LW))
        draw_legend_row(fig, legend_items, LEGEND_Y, GRID['left'], x_max=GRID['right'],
                        dy=LEGEND_DY)

        # Each figure covers a single occlusion level, stated once here
        fig.text(0.5 * (GRID['left'] + GRID['right']), OCCLUSION_TAG_Y, f'{level}% occlusion',
                 fontsize=fs.FS_LEGEND, color=MID, ha='center', va='center', style='italic')

        col0_axes = []
        col_title_x = []

        for row_idx, (model_name, mech_label, mech_short, mech_color, mech_lw, mech_marker) \
                in enumerate(MECHANISMS):
            is_last_row = (row_idx == n_rows - 1)

            for col_idx, (mod_roi, meg_roi) in enumerate(ROI_PAIRS):
                cell = outer[row_idx, col_idx].subgridspec(
                    2, 1, height_ratios=[1.0, RAIL_RATIO], hspace=0.05)
                ax = fig.add_subplot(cell[0])
                rail = fig.add_subplot(cell[1], sharex=ax)
                if col_idx == 0:
                    col0_axes.append(ax)
                if row_idx == 0:
                    col_title_x.append(ax.get_position().x0)

                cond_traces = {}
                cond_vs_zero_idx = {}
                ledger = []
                y_lo, y_hi = 0.0, 0.0

                # (label, colour, linewidth, linestyle, marker)
                cond_style = {
                    'nomask': ('No mask', mech_color, mech_lw, 'solid', mech_marker),
                    'mask':   ('Mask',    MASK_COLOR, MASK_LW, 'solid', 'v'),
                }

                for mask_cond in mask_conditions:
                    X_full = load_rsa_data(model_name, mod_roi, meg_roi, level, mask_cond)
                    if X_full is None:
                        continue

                    baseline_means = np.mean(X_full[:, baseline_mask], axis=1, keepdims=True)
                    X = (X_full - baseline_means)[:, plot_mask]
                    cond_traces[mask_cond] = X

                    peaks_loso, _ = get_loso_latencies(X, plot_times, peak_mask,
                                                       SIGMA_SMOOTH, baseline_val=0)
                    peak_mean = np.mean(peaks_loso)
                    peak_sd = np.std(peaks_loso, ddof=1)

                    onsets_loso, _, _ = get_jackknife_onset_latencies(
                        X, plot_times, stat_mask, alpha=ALPHA,
                        n_permutations=N_PERMUTATIONS, tail=0)

                    for i_sub in range(n_subjects):
                        loso_peaks_records.append({
                            'Subject': sub_ids[i_sub], 'Mechanism': mech_label,
                            'ROI': ROI_DISPLAY[mod_roi], 'Level': level, 'Mask': mask_cond,
                            'Value': peaks_loso[i_sub] * 1000
                        })
                        loso_onsets_records.append({
                            'Subject': sub_ids[i_sub], 'Mechanism': mech_label,
                            'ROI': ROI_DISPLAY[mod_roi], 'Level': level, 'Mask': mask_cond,
                            'Value': onsets_loso[i_sub] * 1000
                        })

                    short, color, lw, ls, marker = cond_style[mask_cond]
                    mean_smooth = gaussian_filter1d(np.mean(X, axis=0), sigma=SIGMA_SMOOTH)
                    sem_smooth = gaussian_filter1d(sem(X, axis=0), sigma=SIGMA_SMOOTH)

                    y_hi = max(y_hi, np.max(mean_smooth + sem_smooth))
                    y_lo = min(y_lo, np.min(mean_smooth - sem_smooth))

                    z = 6 if mask_cond == 'nomask' else 4
                    ax.fill_between(plot_times, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                                    color=color, alpha=0.14, linewidth=0, zorder=z - 2)
                    ax.plot(plot_times, mean_smooth, color=color, linewidth=lw, linestyle=ls,
                            solid_capstyle='round', zorder=z)

                    # Redraw clusters significantly above zero in a heavier line
                    sig_idx = []
                    vs_zero_idx = set()
                    for idx, _p in run_clusters(X, t_threshold, stat_mask, ALPHA, N_PERMUTATIONS):
                        ax.plot(plot_times[idx], mean_smooth[idx], color=color,
                                linewidth=fs.LW_TRACE_SIG, linestyle=ls,
                                solid_capstyle='round', zorder=z)
                        sig_idx.append(idx)
                        vs_zero_idx.update(idx.tolist())
                    cond_vs_zero_idx[mask_cond] = vs_zero_idx

                    # Report the peak only if it lies inside a significant cluster
                    t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
                    if sig_idx and t_idx in set(np.concatenate(sig_idx).tolist()):
                        ax.plot(peak_mean, mean_smooth[t_idx], marker=marker, color=color,
                                markersize=fs.MS_PEAK_SM, zorder=20, markeredgecolor='white',
                                markeredgewidth=fs.pt(0.65), linestyle='none')
                        ledger.append((short, color, marker, peak_mean * 1000, peak_sd * 1000))

                # No-mask vs mask difference, winner-coloured
                lane_clusters = []
                if 'nomask' in cond_traces and 'mask' in cond_traces:
                    # Only report a difference where at least one condition differs from zero
                    vs_zero_idx = (cond_vs_zero_idx.get('nomask', set())
                                   | cond_vs_zero_idx.get('mask', set()))

                    data_diff = cond_traces['nomask'] - cond_traces['mask']
                    raw_diff = data_diff.mean(axis=0)
                    for idx, p in run_clusters(data_diff, t_threshold, stat_mask,
                                               ALPHA, N_PERMUTATIONS):
                        if vs_zero_idx.isdisjoint(idx.tolist()):
                            continue
                        win_is_nomask = bool(raw_diff[idx].mean() > 0)
                        lane_clusters.append((idx, win_is_nomask))
                        sig_windows_records.append({
                            'Mechanism': mech_label, 'ROI': ROI_DISPLAY[mod_roi], 'Level': level,
                            'Comparison': 'No Mask vs Mask',
                            'Winner': 'No Mask' if win_is_nomask else 'Mask',
                            'Loser': 'Mask' if win_is_nomask else 'No Mask',
                            'Start_Time_ms': plot_times[idx[0]] * 1000,
                            'End_Time_ms': plot_times[idx[-1]] * 1000,
                            'P_Value': p, 'Stat_Window': '0-400ms'
                        })

                span = max(y_hi - y_lo, 1e-6)
                ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.06 * span)
                ax.axhline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, zorder=1)
                ax.axvline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE,
                           linestyle=(0, (1, 2.4)), zorder=1)
                style_axes(ax)
                style_time_axis(ax)

                draw_ledger(fig, ax, ledger)

                # One comparison per panel; the legend fixes the pair, so the rail needs no label
                draw_rail_single(rail, lane_clusters, mech_color, MASK_COLOR,
                                 plot_times, TMIN, TMAX)
                style_time_axis(rail, ticklabels=is_last_row, label=is_last_row)

                if col_idx == 0:
                    ax.set_ylabel('Spearman ρ', fontsize=fs.FS_AXLABEL, color=INK,
                                  labelpad=fs.pt(1.8))

        for x0, (mod_roi, _) in zip(col_title_x, ROI_PAIRS):
            fig.text(x0, COLTITLE_Y, f'{ROI_DISPLAY[mod_roi]} / {LAYER_DISPLAY[mod_roi]}',
                     fontsize=fs.FS_COLTITLE,
                     fontweight='bold', color=INK, ha='left', va='bottom')

        draw_cascade(fig, col0_axes,
                     ROW_LABELS,
                     [m[3] for m in MECHANISMS],
                     markers=[m[5] for m in MECHANISMS])

        save_loso_csv(loso_peaks_records,
                      os.path.join(out_dir, f'MaskEffect_LOSO_Peak_Latencies_Occ{level}.csv'))
        save_loso_csv(loso_onsets_records,
                      os.path.join(out_dir, f'MaskEffect_LOSO_Onset_Latencies_Occ{level}.csv'))
        # Always written (header-only if nothing is significant) so an old file is never left behind
        pd.DataFrame(sig_windows_records, columns=SIG_WINDOW_COLUMNS).to_csv(
            os.path.join(out_dir, f'MaskEffect_Significance_Windows_Occ{level}.csv'), index=False)

        paths = save_figure(fig, out_dir, f'Figure_MaskEffect_Occ{level}')
        report_page_fit(FIGSIZE, f'Figure_MaskEffect_Occ{level}')
        plt.close(fig)
        log.info('Saved %s', paths[0])


if __name__ == '__main__':
    main()
