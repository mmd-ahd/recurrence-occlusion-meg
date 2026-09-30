"""Model-brain RSA time courses: HTRN (Fig. 5), BLT-VS (Supp. Figs. 8-9) or CORnet (Supp. Figs. 10-11).

Spearman correlation between MEG ROI RDMs and model RDMs over time, for 0% and 60% occlusion (one
figure per mask condition). Curves show the mean over subjects with SEM; thick segments mark
clusters significantly above baseline (cluster-permutation test on 0-400 ms). Pairwise differences
between model variants are drawn on a rail below each panel in the colour of the winner, and
leave-one-subject-out peak latencies (mean ± SD) are listed above each panel.

Models:
    htrn    HTRN-FF / -LR / -TD (readout stages 0 / 6 / 7 of one network)
    blt     BLT-VS B / BL / BLT (feed-forward / lateral / lateral + top-down)
    cornet  CORnet-Z vs CORnet-RT (one trace per architecture, no top-down pathway)

Inputs:  Megocclusion/derivatives/RSA_Subaveraged_Results_<ResNet|BLT|CORNet>/ (``compute_rsa.py``)
Outputs: figure (PNG/SVG/PDF), latency CSVs and significance-window CSVs in that folder's
         ``plots_Mechanism_Comparison*_v2/``

Usage:
    python scripts/07_figures/plot_rsa_variants.py --model htrn
"""

import argparse
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
    INK, MID, HAIRLINE, GROUND,
    get_loso_latencies, get_jackknife_onset_latencies, save_loso_csv, run_clusters,
    style_axes, draw_legend_row, draw_ledger, draw_rail_multi, draw_cascade,
    save_figure, report_page_fit,
)

log = logging.getLogger(__name__)
fs.setup_rcparams()

derivatives_dir = str(DERIVATIVES_DIR)
meg_pattern_dir = os.path.join(derivatives_dir, 'Patterns_MEG')

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

# Per model: RSA results folder, plot folder, figure name stem, model layers compared with each ROI
# (column titles), the series as (RSA model name, label, short label, colour, linewidth, peak
# marker, peak-ledger label), the pairwise comparisons (top lane first) and whether onset
# latencies are exported. Marker and line weight repeat the colour coding so the series stay
# distinguishable for colour-blind readers.
MODELS = {
    'htrn': dict(
        rsa_folder='RSA_Subaveraged_Results_ResNet',
        plot_folder='plots_Mechanism_Comparison_v2',
        stem='Figure_Mechanism',
        layer_display={'V1_3': 'Conv2-3', 'LOC': 'Conv4', 'IT': 'Conv5'},
        title_suffix='',
        series=[
            ('DFM-Feedforward',     'Feedforward (FF)',     'FF', '#F4256D', fs.pt(1.30), 'o', 'FF'),
            ('DFM-Recurrent',       'Local-Recurrent (LR)', 'LR', '#3A0CA3', fs.pt(1.50), 's', 'LR'),
            ('DFM-TopDownFeedback', 'Top-Down (TD)',        'TD', '#00A5A8', fs.pt(1.50), 'D', 'TD'),
        ],
        diff_pairs=[(0, 1), (1, 2), (0, 2)],
        onsets=True,
    ),
    'blt': dict(
        rsa_folder='RSA_Subaveraged_Results_BLT',
        plot_folder='plots_Mechanism_Comparison_BLT_v2',
        stem='Figure_Mechanism_BLT',
        # BLT-VS area names; its "LOC" is the IT-equivalent top area, not the lateral occipital ROI
        layer_display={'V1_3': 'V1-V3', 'LOC': 'V4', 'IT': 'LOC'},
        title_suffix=' (model)',
        series=[
            ('BLT-Feedforward',     'BLT-FF (B)',   'BLT-FF', '#E8A200', fs.pt(1.30), 'o', 'B'),
            ('BLT-Recurrent',       'BLT-LR (BL)',  'BLT-LR', '#C0369D', fs.pt(1.50), 's', 'BL'),
            ('BLT-TopDownFeedback', 'BLT-TD (BLT)', 'BLT-TD', '#0EA5C4', fs.pt(1.50), 'D', 'BLT'),
        ],
        diff_pairs=[(0, 1), (1, 2), (0, 2)],
        onsets=False,
    ),
    'cornet': dict(
        rsa_folder='RSA_Subaveraged_Results_CORNet',
        plot_folder='plots_Mechanism_Comparison_CORNet_v2',
        stem='Figure_Mechanism_CORNet',
        layer_display={'V1_3': 'V1-V2', 'LOC': 'V4', 'IT': 'IT'},
        title_suffix=' (model)',
        series=[
            ('CORNet-Z-Feedforward', 'Feedforward (CORnet-Z)',     'Z',  '#2A78D6', fs.pt(1.30), 'o', 'Z'),
            ('CORNet-RT-Recurrent',  'Local-Recurrent (CORnet-RT)', 'RT', '#EB6834', fs.pt(1.50), 's', 'RT'),
        ],
        diff_pairs=[(0, 1)],
        onsets=False,
    ),
}

# The axes start at -50 ms, but tests and the peak search use t >= 0 only
TMIN, TMAX = -0.05, 0.4
TICK_STEP = 0.05
ALPHA = 0.05
SIGMA_SMOOTH = 2.0
N_PERMUTATIONS = 'all'

# Layout: one row per occlusion level, one column per ROI
FIGSIZE = (fs.FIG_W, 9.10)
GRID = dict(hspace=0.13, wspace=0.13, left=0.098, right=0.988, top=0.850, bottom=0.065)
LEGEND_Y = 0.986
COLTITLE_Y = 0.908


def style_time_axis(ax, ticklabels=False, label=False):
    """Time axis with this figure's limits and tick spacing."""
    fs.style_time_axis(ax, TMIN, TMAX, TICK_STEP, ticklabels=ticklabels, label=label)


SIG_WINDOW_COLUMNS = ['ROI', 'Level', 'Comparison', 'Winner', 'Loser',
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


def load_rsa_data(rsa_dir, model_name, mod_roi, meg_roi, level, mask_cond):
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
    """Draw one figure per mask condition and save the CSVs."""
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()

    setup_logging()
    cfg = MODELS[args.model]
    MECHANISMS = cfg['series']
    DIFF_PAIRS = cfg['diff_pairs']
    LAYER_DISPLAY = cfg['layer_display']
    RAIL_RATIO = 0.10 + 0.06 * len(DIFF_PAIRS)
    rsa_dir = os.path.join(derivatives_dir, cfg['rsa_folder'])
    out_dir = os.path.join(rsa_dir, cfg['plot_folder'])
    os.makedirs(out_dir, exist_ok=True)

    full_times = load_meg_time_vector()
    baseline_mask = full_times < 0
    plot_mask = (full_times >= TMIN) & (full_times <= TMAX)
    plot_times = full_times[plot_mask]

    n_subjects = len(sub_ids)
    t_threshold = stats.t.ppf(1 - ALPHA / 2, n_subjects - 1)
    peak_mask = plot_times >= 0.0
    stat_mask = plot_times >= 0.0
    n_rows = len(occlusion_levels)

    for mask_cond in mask_conditions:
        log.info('Mask condition: %s', mask_cond)
        loso_peaks_records = []
        loso_onsets_records = []
        sig_windows_records = []

        fig = plt.figure(figsize=FIGSIZE, dpi=110, facecolor=GROUND)
        outer = fig.add_gridspec(n_rows, len(ROI_PAIRS), **GRID)

        legend_items = [
            (label, color, marker, fs.MS_LEGEND if marker != 'D' else fs.MS_LEGEND_SM, None, 0)
            for (_, label, short, color, lw, marker, _ledger) in MECHANISMS
        ]
        draw_legend_row(fig, legend_items, LEGEND_Y, GRID['left'])

        # Only the mask condition is labelled; no-mask is the default
        if mask_cond == 'mask':
            fig.text(0.5 * (GRID['left'] + GRID['right']), 0.954, 'Mask condition',
                     fontsize=fs.FS_LEGEND, color=MID, ha='center', va='center', style='italic')

        col0_axes = []
        col_title_x = []

        for row_idx, level in enumerate(occlusion_levels):
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

                mech_traces = [None] * len(MECHANISMS)
                mech_vs_zero_idx = [set() for _ in MECHANISMS]
                ledger = []
                y_lo, y_hi = 0.0, 0.0

                for m_idx, (model_name, label, short, color, lw, marker,
                            ledger_short) in enumerate(MECHANISMS):
                    X_full = load_rsa_data(rsa_dir, model_name, mod_roi, meg_roi, level, mask_cond)
                    if X_full is None:
                        continue

                    baseline_means = np.mean(X_full[:, baseline_mask], axis=1, keepdims=True)
                    X = (X_full - baseline_means)[:, plot_mask]
                    mech_traces[m_idx] = X

                    peaks_loso, _ = get_loso_latencies(X, plot_times, peak_mask,
                                                       SIGMA_SMOOTH, baseline_val=0)
                    peak_mean = np.mean(peaks_loso)
                    peak_sd = np.std(peaks_loso, ddof=1)

                    if cfg['onsets']:
                        onsets_loso, _, _ = get_jackknife_onset_latencies(
                            X, plot_times, stat_mask, alpha=ALPHA,
                            n_permutations=N_PERMUTATIONS, tail=0)

                    for i_sub in range(n_subjects):
                        loso_peaks_records.append({
                            'Subject': sub_ids[i_sub], 'ROI': ROI_DISPLAY[mod_roi], 'Level': level,
                            'Mechanism': label, 'Mask': mask_cond,
                            'Value': peaks_loso[i_sub] * 1000
                        })
                        if cfg['onsets']:
                            loso_onsets_records.append({
                                'Subject': sub_ids[i_sub], 'ROI': ROI_DISPLAY[mod_roi],
                                'Level': level, 'Mechanism': label, 'Mask': mask_cond,
                                'Value': onsets_loso[i_sub] * 1000
                            })

                    mean_smooth = gaussian_filter1d(np.mean(X, axis=0), sigma=SIGMA_SMOOTH)
                    sem_smooth = gaussian_filter1d(sem(X, axis=0), sigma=SIGMA_SMOOTH)

                    y_hi = max(y_hi, np.max(mean_smooth + sem_smooth))
                    y_lo = min(y_lo, np.min(mean_smooth - sem_smooth))

                    ax.fill_between(plot_times, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                                    color=color, alpha=0.14, linewidth=0, zorder=2 + m_idx)
                    ax.plot(plot_times, mean_smooth, color=color, linewidth=lw,
                            solid_capstyle='round', zorder=6 + m_idx)

                    # Redraw clusters significantly above zero in a heavier line
                    sig_idx = []
                    for idx, _p in run_clusters(X, t_threshold, stat_mask, ALPHA, N_PERMUTATIONS):
                        ax.plot(plot_times[idx], mean_smooth[idx], color=color,
                                linewidth=fs.LW_TRACE_SIG, solid_capstyle='round', zorder=6 + m_idx)
                        sig_idx.append(idx)
                        mech_vs_zero_idx[m_idx].update(idx.tolist())

                    # Report the peak only if it lies inside a significant cluster
                    t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
                    if sig_idx and t_idx in set(np.concatenate(sig_idx).tolist()):
                        ax.plot(peak_mean, mean_smooth[t_idx], marker=marker, color=color,
                                markersize=fs.MS_PEAK if marker != 'D' else fs.MS_PEAK_SM,
                                zorder=20, markeredgecolor='white',
                                markeredgewidth=fs.pt(0.65), linestyle='none')
                        ledger.append((ledger_short, color, marker, peak_mean * 1000, peak_sd * 1000))

                # Pairwise differences between variants, winner-coloured
                lanes = []
                for a, b in DIFF_PAIRS:
                    sa, ca = MECHANISMS[a][2], MECHANISMS[a][3]
                    sb, cb = MECHANISMS[b][2], MECHANISMS[b][3]
                    if mech_traces[a] is None or mech_traces[b] is None:
                        lanes.append((sa, ca, sb, cb, []))
                        continue

                    # Only report a difference where at least one variant differs from zero
                    vs_zero_idx = mech_vs_zero_idx[a] | mech_vs_zero_idx[b]

                    data_diff = mech_traces[a] - mech_traces[b]
                    mean_diff = data_diff.mean(axis=0)
                    lane_clusters = []
                    for idx, p in run_clusters(data_diff, t_threshold, stat_mask,
                                               ALPHA, N_PERMUTATIONS):
                        if vs_zero_idx.isdisjoint(idx.tolist()):
                            continue
                        win_is_a = bool(mean_diff[idx].mean() > 0)
                        lane_clusters.append((idx, p, win_is_a))
                        sig_windows_records.append({
                            'ROI': ROI_DISPLAY[mod_roi], 'Level': level,
                            'Comparison': f'{MECHANISMS[a][1]} vs {MECHANISMS[b][1]}',
                            'Winner': MECHANISMS[a if win_is_a else b][1],
                            'Loser': MECHANISMS[b if win_is_a else a][1],
                            'Start_Time_ms': plot_times[idx[0]] * 1000,
                            'End_Time_ms': plot_times[idx[-1]] * 1000,
                            'P_Value': p, 'Stat_Window': '0-400ms'
                        })
                    lanes.append((sa, ca, sb, cb, lane_clusters))

                span = max(y_hi - y_lo, 1e-6)
                ax.set_ylim(y_lo - 0.06 * span, y_hi + 0.06 * span)
                ax.axhline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, zorder=1)
                ax.axvline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE,
                           linestyle=(0, (1, 2.4)), zorder=1)
                style_axes(ax)
                style_time_axis(ax)

                draw_ledger(fig, ax, ledger)

                draw_rail_multi(fig, rail, lanes, plot_times, TMIN, TMAX)
                style_time_axis(rail, ticklabels=is_last_row, label=is_last_row)

                if col_idx == 0:
                    ax.set_ylabel('Spearman ρ', fontsize=fs.FS_AXLABEL, color=INK,
                                  labelpad=fs.pt(1.8))

        # Column titles: ROI and the model layer compared with it
        for x0, (mod_roi, _meg_roi) in zip(col_title_x, ROI_PAIRS):
            fig.text(x0, COLTITLE_Y, f'{ROI_DISPLAY[mod_roi]} / {LAYER_DISPLAY[mod_roi]}{cfg["title_suffix"]}',
                     fontsize=fs.FS_COLTITLE, fontweight='bold', color=INK, ha='left', va='bottom')

        draw_cascade(fig, col0_axes,
                     [f'{level}% occlusion' for level in occlusion_levels],
                     [INK] * len(occlusion_levels))

        save_loso_csv(loso_peaks_records,
                      os.path.join(out_dir, f'Mechanism_LOSO_Peak_Latencies_{mask_cond}.csv'))
        if cfg['onsets']:
            save_loso_csv(loso_onsets_records,
                          os.path.join(out_dir, f'Mechanism_LOSO_Onset_Latencies_{mask_cond}.csv'))
        # Always written (header-only if nothing is significant) so an old file is never left behind
        pd.DataFrame(sig_windows_records, columns=SIG_WINDOW_COLUMNS).to_csv(
            os.path.join(out_dir, f'Mechanism_Significance_Windows_{mask_cond}.csv'), index=False)

        paths = save_figure(fig, out_dir, f"{cfg['stem']}_{mask_cond}")
        report_page_fit(FIGSIZE, f"{cfg['stem']}_{mask_cond}")
        plt.close(fig)
        log.info('Saved %s', paths[0])


if __name__ == '__main__':
    main()
