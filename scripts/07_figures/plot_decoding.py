"""Category decoding time courses (Fig. 1B and Fig. 2).

Figure A (``Figure_Decoding_ROIComparison_v2``, Fig. 1B): pairwise category decoding accuracy in
V1-3, LOC and IT-PHC for unmasked trials at 0% and 60% occlusion, with a rail of pairwise ROI
differences below each panel.

Figure B (``Figure_Decoding_MaskComparison_v2``, Fig. 2): for each ROI and occlusion level, decoders
trained and tested on unmasked trials ("within") against decoders trained on masked and tested on
unmasked trials ("cross"), with the within-minus-cross difference and its significance rail.

Curves show the mean over subjects with SEM; thick segments are clusters significantly above
chance (cluster-permutation test, t >= 0). Peak latencies are leave-one-subject-out means ± SD.

Inputs:  Megocclusion/derivatives/decoding/ and decoding_cross_condition/
         (``decode_categories.py``, ``decode_cross_mask.py``)
Outputs: both figures (PNG/SVG/PDF) plus ``Decoding_v2_LOSO_Peak_Latencies.csv`` and
         ``Decoding_v2_Significance_Windows.csv`` in ``decoding/plots_Decoding_v2/``

Usage:
    python scripts/07_figures/plot_decoding.py
"""

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
    INK, HAIRLINE, GROUND, ROI_COLORS,
    get_loso_latencies, save_loso_csv, run_clusters, filter_diff_clusters,
    style_axes, draw_legend_row, draw_ledger, draw_rail_single, draw_rail_multi,
    draw_diff_panel, draw_cascade, save_figure, report_page_fit,
)

log = logging.getLogger(__name__)
fs.setup_rcparams()

derivatives_dir = str(DERIVATIVES_DIR)
decoding_dir = os.path.join(derivatives_dir, 'decoding')
cross_decoding_dir = os.path.join(derivatives_dir, 'decoding_cross_condition')
out_dir = os.path.join(decoding_dir, 'plots_Decoding_v2')

sub_ids = SUBJECT_IDS
occlusion_levels = ['0', '60']

# Figure A series (roi, label, short label, colour, linewidth, marker)
ROI_SERIES = [
    ('V1-3',   'V1-3',   'V1-3', ROI_COLORS['V1-3'],   fs.pt(1.45), 'o'),
    ('LOC',    'LOC',    'LOC',  ROI_COLORS['LOC'],    fs.pt(1.65), 's'),
    ('IT-PHC', 'IT-PHC', 'IT',   ROI_COLORS['IT-PHC'], fs.pt(1.65), 'D'),
]
ROI_DIFF_PAIRS = [(0, 1), (1, 2), (0, 2)]

# Figure B series: within (no mask -> no mask) vs cross (mask -> no mask). The difference panel
# expects (within, cross) in this order: positive = within ahead.
WC_SERIES = [
    ('within', 'Within (no mask → no mask)', 'Within', '#4C3B8C', fs.pt(1.55), '^'),
    ('cross',  'Cross (mask → no mask)',      'Cross',  '#FF6B35', fs.pt(1.55), 'v'),
]
WC_DIFF_PAIRS = [(0, 1)]

DIFF_PANEL_RATIO = 0.40   # height of the difference panel relative to the trace panel
# Headroom of the difference panel's ceiling above its largest value; must exceed ~1.32 so the
# peak does not run through the ledger text
DIFF_HEADROOM = 1.6

# The axes start at -50 ms, but tests and the peak search use t >= 0 only. Baseline is chance.
TMIN, TMAX = -0.05, 0.5
TICK_STEP = 0.05
ALPHA = 0.05
SIGMA_SMOOTH = 2.0
N_PERMUTATIONS = 'all'
CHANCE = 0.5
Y_FLOOR = 45.0  # accuracy axis floor (%)

# Layout (each figure is sized to its own rows and columns)
ROI_RAIL_RATIO = 0.10 + 0.06 * len(ROI_DIFF_PAIRS)
WC_RAIL_RATIO = 0.10 + 0.06 * len(WC_DIFF_PAIRS)

# Figure A: one row x two occlusion-level columns
ROI_FIGSIZE = (fs.FIG_W, 5.20)
ROI_GRID = dict(wspace=0.13, left=0.075, right=0.988, top=0.84, bottom=0.115)
ROI_LEGEND_Y = 0.965
ROI_COLTITLE_Y = 0.89

# Figure B: two occlusion-level rows x three ROI columns; each cell has trace, difference and rail
WC_FIGSIZE = (fs.FIG_W, 9.60)
WC_GRID = dict(hspace=0.16, wspace=0.15, left=0.098, right=0.988, top=0.89, bottom=0.075)
WC_LEGEND_Y = 0.981
WC_COLTITLE_Y = 0.926


def style_time_axis(ax, ticklabels=False, label=False):
    """Time axis with this figure's limits and tick spacing."""
    fs.style_time_axis(ax, TMIN, TMAX, TICK_STEP, ticklabels=ticklabels, label=label)


def thin_time_ticks(ax, keep_ms=(-50, 0, 100, 200, 300, 400, 500)):
    """Keep tick marks but label only the times in `keep_ms` (narrow panels)."""
    keep = set(keep_ms)
    ax.set_xticklabels([f'{round(t * 1000):.0f}' if round(t * 1000) in keep else ''
                        for t in ax.get_xticks()])


def load_meg_time_vector():
    """Time vector of the decoding scores (default -100 to 600 ms if none is found)."""
    for candidate in (
        os.path.join(decoding_dir, 'time_centers.npy'),
        os.path.join(decoding_dir, 'time_points.npy'),
        os.path.join(cross_decoding_dir, 'time_points.npy'),
    ):
        if os.path.exists(candidate):
            return np.load(candidate)
    return np.linspace(-0.1, 0.6, 176)


def load_within(roi, level):
    """(subjects, times) accuracy trained and tested on unmasked trials; None if a subject is missing."""
    group_data = []
    for sub in sub_ids:
        fpath = os.path.join(decoding_dir, f'sub-{sub:02d}', level, f'scores_{roi}_avg-pairs.npy')
        if os.path.exists(fpath):
            group_data.append(np.load(fpath))
    if len(group_data) < len(sub_ids):
        return None
    return np.array(group_data)


def load_cross(roi, level):
    """(subjects, times) accuracy trained on masked, tested on unmasked trials; None if a subject is missing."""
    group_data = []
    for sub in sub_ids:
        fpath = os.path.join(cross_decoding_dir, f'sub-{sub:02d}',
                             f'train-mask_test-nomask_{level}', f'scores_{roi}_avg-pairs.npy')
        if os.path.exists(fpath):
            group_data.append(np.load(fpath))
    if len(group_data) < len(sub_ids):
        return None
    return np.array(group_data)


def draw_panel(fig, ax, ax_diff, rail, series_defs, loaders, level, n_subjects, plot_times,
               plot_mask, peak_mask, stat_mask, t_threshold, diff_pairs, loso_peaks_records,
               sig_windows_records, row_label, row_type):
    """Draw one trace panel with its significance rail and, optionally, a difference panel.

    Returns (padded_upper, twin_abs_max): the panel's own y ceiling and the difference panel's
    maximum |value|, so callers can share ceilings across panels before finalising the limits.
    """
    traces = [None] * len(series_defs)
    vs_zero_idx = [set() for _ in series_defs]
    ledger = []
    y_hi = 0.0
    twin_abs_max = None

    for s_idx, (key, label, short, color, lw, marker) in enumerate(series_defs):
        X_full = loaders[key](level)
        if X_full is None:
            continue

        X = X_full[:, plot_mask] - CHANCE
        traces[s_idx] = X

        peaks_loso, _ = get_loso_latencies(X, plot_times, peak_mask, SIGMA_SMOOTH, baseline_val=0)
        peak_mean = np.mean(peaks_loso)
        peak_sd = np.std(peaks_loso, ddof=1)

        for i_sub in range(n_subjects):
            loso_peaks_records.append({
                'Subject': sub_ids[i_sub], 'RowType': row_type, 'Row': row_label, 'Level': level,
                'Series': label, 'Value': peaks_loso[i_sub] * 1000
            })

        mean_smooth = gaussian_filter1d(np.mean(X, axis=0), sigma=SIGMA_SMOOTH)
        sem_smooth = gaussian_filter1d(sem(X, axis=0), sigma=SIGMA_SMOOTH)

        plot_mean = (mean_smooth + CHANCE) * 100
        plot_sem = sem_smooth * 100

        y_hi = max(y_hi, np.max(plot_mean + plot_sem))

        ax.fill_between(plot_times, plot_mean - plot_sem, plot_mean + plot_sem,
                        color=color, alpha=0.14, linewidth=0, zorder=2 + s_idx)
        ax.plot(plot_times, plot_mean, color=color, linewidth=lw,
                solid_capstyle='round', zorder=6 + s_idx)

        # Redraw clusters significantly above chance in a heavier line
        sig_idx = []
        for idx, _p in run_clusters(X, t_threshold, stat_mask, ALPHA, N_PERMUTATIONS):
            ax.plot(plot_times[idx], plot_mean[idx], color=color, linewidth=fs.LW_TRACE_SIG,
                    solid_capstyle='round', zorder=6 + s_idx)
            sig_idx.append(idx)
            vs_zero_idx[s_idx].update(idx.tolist())

        # Report the peak only if it lies inside a significant cluster
        t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
        if sig_idx and t_idx in set(np.concatenate(sig_idx).tolist()):
            ax.plot(peak_mean, plot_mean[t_idx], marker=marker, color=color,
                    markersize=fs.MS_PEAK if marker != 'D' else fs.MS_PEAK_SM, zorder=20,
                    markeredgecolor='white', markeredgewidth=fs.pt(0.70), linestyle='none')
            ledger.append((short, color, marker, peak_mean * 1000, peak_sd * 1000))

    lane_data = []
    for a, b in diff_pairs:
        sa, ca = series_defs[a][2], series_defs[a][3]
        sb, cb = series_defs[b][2], series_defs[b][3]
        if traces[a] is None or traces[b] is None:
            lane_data.append((sa, ca, sb, cb, []))
            continue

        data_diff = traces[a] - traces[b]
        mean_diff = data_diff.mean(axis=0)
        raw_clusters = run_clusters(data_diff, t_threshold, stat_mask, ALPHA, N_PERMUTATIONS)
        lane_clusters = filter_diff_clusters(raw_clusters, vs_zero_idx[a], vs_zero_idx[b],
                                             mean_diff)
        for idx, p, win_is_a in lane_clusters:
            sig_windows_records.append({
                'RowType': row_type, 'Row': row_label, 'Level': level,
                'Comparison': f'{series_defs[a][1]} vs {series_defs[b][1]}',
                'Winner': series_defs[a if win_is_a else b][1],
                'Loser': series_defs[b if win_is_a else a][1],
                'Start_Time_ms': plot_times[idx[0]] * 1000,
                'End_Time_ms': plot_times[idx[-1]] * 1000,
                'P_Value': p, 'Stat_Window': f'0-{int(TMAX * 1000)}ms'
            })
        lane_data.append((sa, ca, sb, cb, lane_clusters))

        # Within-minus-cross difference panel (figure B), reusing the statistics above
        if ax_diff is not None:
            twin_abs_max = draw_diff_panel(
                fig, ax_diff, plot_times, data_diff, lane_clusters, ca, cb,
                peak_mask, SIGMA_SMOOTH, loso_peaks_records,
                {'RowType': row_type, 'Row': row_label, 'Level': level,
                 'Series': 'Difference (Within − Cross)'},
                sub_ids, 'Difference', TMIN, TMAX, TICK_STEP, scale=100.0)

    span = max(y_hi - Y_FLOOR, 1e-6)
    padded_upper = y_hi + 0.06 * span
    ax.set_ylim(Y_FLOOR, padded_upper)
    ax.axhline(50, color=HAIRLINE, linewidth=fs.LW_GUIDE, zorder=1)
    ax.axvline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, linestyle=(0, (1, 2.4)), zorder=1)
    style_axes(ax)
    style_time_axis(ax)

    draw_ledger(fig, ax, ledger)

    if max(len(diff_pairs), 1) >= 2:
        draw_rail_multi(fig, rail, lane_data, plot_times, TMIN, TMAX)
    else:
        sa, ca, sb, cb, clusters = lane_data[0]
        draw_rail_single(rail, [(idx, w) for idx, _p, w in clusters], ca, cb,
                         plot_times, TMIN, TMAX)

    return padded_upper, twin_abs_max


def build_roi_comparison_figure(plot_times, plot_mask, peak_mask, stat_mask, t_threshold,
                                n_subjects, loso_peaks_records, sig_windows_records):
    """Figure A: ROIs overlaid for unmasked trials; one row, one column per occlusion level."""
    loaders_roi = {roi: (lambda level, r=roi: load_within(r, level)) for roi, *_ in ROI_SERIES}

    fig = plt.figure(figsize=ROI_FIGSIZE, dpi=110, facecolor=GROUND)
    outer = fig.add_gridspec(1, len(occlusion_levels),
                             height_ratios=[1.0 + ROI_RAIL_RATIO], **ROI_GRID)

    legend_items = [(label, color, marker, fs.MS_LEGEND if marker != 'D' else fs.MS_LEGEND_SM,
                     None, 0) for (_, label, short, color, lw, marker) in ROI_SERIES]
    draw_legend_row(fig, legend_items, ROI_LEGEND_Y, ROI_GRID['left'], x_max=ROI_GRID['right'])

    row_entries = []  # (ax, padded_upper); the y ceiling is shared across the two columns
    col_title_x = []
    for col_idx, level in enumerate(occlusion_levels):
        cell = outer[0, col_idx].subgridspec(
            2, 1, height_ratios=[1.0, ROI_RAIL_RATIO], hspace=0.05)
        ax = fig.add_subplot(cell[0])
        rail = fig.add_subplot(cell[1], sharex=ax)
        col_title_x.append(ax.get_position().x0)

        padded_upper, _twin = draw_panel(
            fig, ax, None, rail, ROI_SERIES, loaders_roi, level, n_subjects, plot_times,
            plot_mask, peak_mask, stat_mask, t_threshold, ROI_DIFF_PAIRS,
            loso_peaks_records, sig_windows_records,
            row_label='ROI comparison', row_type='ROI-Comparison')
        row_entries.append((ax, padded_upper))

        style_time_axis(rail, ticklabels=True, label=True)
        ax.set_ylabel('Accuracy (%)', fontsize=fs.FS_AXLABEL, color=INK, labelpad=fs.pt(1.8))

    shared_upper = max(pu for _ax, pu in row_entries)
    for ax, _pu in row_entries:
        ax.set_ylim(Y_FLOOR, shared_upper)

    for x0, level in zip(col_title_x, occlusion_levels):
        fig.text(x0, ROI_COLTITLE_Y, f'{level}% occlusion', fontsize=fs.FS_COLTITLE,
                 fontweight='bold', color=INK, ha='left', va='bottom')

    paths = save_figure(fig, out_dir, 'Figure_Decoding_ROIComparison_v2')
    report_page_fit(ROI_FIGSIZE, 'Figure_Decoding_ROIComparison_v2')
    plt.close(fig)
    log.info('Saved %s', paths[0])


def build_mask_comparison_figure(plot_times, plot_mask, peak_mask, stat_mask, t_threshold,
                                 n_subjects, loso_peaks_records, sig_windows_records):
    """Figure B: within vs cross decoding per ROI; rows = occlusion level, columns = ROI.

    The y ceiling (and the difference panel's symmetric ceiling) is shared down each ROI column
    so the two occlusion levels can be compared by eye.
    """
    n_rows, n_cols = len(occlusion_levels), len(ROI_SERIES)
    fig = plt.figure(figsize=WC_FIGSIZE, dpi=110, facecolor=GROUND)
    outer = fig.add_gridspec(n_rows, n_cols,
                             height_ratios=[1.0 + DIFF_PANEL_RATIO + WC_RAIL_RATIO] * n_rows,
                             **WC_GRID)

    legend_items = [(label, color, marker, fs.MS_LEGEND, None, 0)
                    for (_, label, short, color, lw, marker) in WC_SERIES]
    draw_legend_row(fig, legend_items, WC_LEGEND_Y, WC_GRID['left'], x_max=WC_GRID['right'])

    col0_axes = []
    col_title_x = []
    # Per column: (ax, ax_diff, padded_upper, twin_max) of both rows, for the shared-ceiling pass
    columns_data = [[] for _ in range(n_cols)]

    for row_idx, level in enumerate(occlusion_levels):
        is_last_row = (row_idx == n_rows - 1)
        for col_idx, (roi, label, short, color, lw, marker) in enumerate(ROI_SERIES):
            loaders = {'within': (lambda lv, r=roi: load_within(r, lv)),
                       'cross':  (lambda lv, r=roi: load_cross(r, lv))}

            cell = outer[row_idx, col_idx].subgridspec(
                3, 1, height_ratios=[1.0, DIFF_PANEL_RATIO, WC_RAIL_RATIO], hspace=0.07)
            ax = fig.add_subplot(cell[0])
            ax_diff = fig.add_subplot(cell[1], sharex=ax)
            rail = fig.add_subplot(cell[2], sharex=ax)

            if col_idx == 0:
                col0_axes.append(ax)
            if row_idx == 0:
                col_title_x.append(ax.get_position().x0)

            padded_upper, twin_max = draw_panel(
                fig, ax, ax_diff, rail, WC_SERIES, loaders, level, n_subjects, plot_times,
                plot_mask, peak_mask, stat_mask, t_threshold, WC_DIFF_PAIRS,
                loso_peaks_records, sig_windows_records,
                row_label=label, row_type='Within-Cross')
            columns_data[col_idx].append((ax, ax_diff, padded_upper, twin_max))

            style_time_axis(rail, ticklabels=is_last_row, label=is_last_row)
            if is_last_row:
                thin_time_ticks(rail)
            if col_idx == 0:
                # Short labels: rotated text must fit the panel height
                ax.set_ylabel('Accuracy (%)', fontsize=fs.FS_AXLABEL, color=INK,
                              labelpad=fs.pt(1.8))
                ax_diff.set_ylabel('Δ accuracy (pp)', fontsize=fs.FS_AXLABEL_SM,
                                   color=INK, labelpad=fs.pt(2.6))

    # Share the y ceiling (and the difference panel's symmetric ceiling) down each ROI column
    for col_entries in columns_data:
        shared_upper = max(pu for (_ax, _ad, pu, _tm) in col_entries)
        for ax, _ad, _pu, _tm in col_entries:
            ax.set_ylim(Y_FLOOR, shared_upper)
        twin_maxes = [tm for (_ax, ad, _pu, tm) in col_entries if ad is not None and tm is not None]
        if twin_maxes:
            shared_twin = max(twin_maxes) * DIFF_HEADROOM
            for _ax, ad, _pu, _tm in col_entries:
                if ad is not None:
                    ad.set_ylim(-shared_twin, shared_twin)

    # Column titles use the ROI colours of figure A
    for x0, (roi, label, short, color, lw, marker) in zip(col_title_x, ROI_SERIES):
        fig.text(x0, WC_COLTITLE_Y, label, fontsize=fs.FS_COLTITLE, fontweight='bold',
                 color=color, ha='left', va='bottom')

    draw_cascade(fig, col0_axes, [f'{level}% occlusion' for level in occlusion_levels],
                 [INK] * n_rows)

    paths = save_figure(fig, out_dir, 'Figure_Decoding_MaskComparison_v2')
    report_page_fit(WC_FIGSIZE, 'Figure_Decoding_MaskComparison_v2')
    plt.close(fig)
    log.info('Saved %s', paths[0])


def main():
    """Draw both decoding figures and save the latency and significance CSVs."""
    setup_logging()
    os.makedirs(out_dir, exist_ok=True)

    full_times = load_meg_time_vector()
    plot_mask = (full_times >= TMIN) & (full_times <= TMAX)
    plot_times = full_times[plot_mask]

    n_subjects = len(sub_ids)
    t_threshold = stats.t.ppf(1 - ALPHA / 2, n_subjects - 1)
    peak_mask = plot_times >= 0.0
    stat_mask = plot_times >= 0.0

    loso_peaks_records = []
    sig_windows_records = []

    build_roi_comparison_figure(plot_times, plot_mask, peak_mask, stat_mask, t_threshold,
                                n_subjects, loso_peaks_records, sig_windows_records)
    build_mask_comparison_figure(plot_times, plot_mask, peak_mask, stat_mask, t_threshold,
                                 n_subjects, loso_peaks_records, sig_windows_records)

    # Both figures share the CSVs; the RowType column tells their records apart
    save_loso_csv(loso_peaks_records, os.path.join(out_dir, 'Decoding_v2_LOSO_Peak_Latencies.csv'))
    if sig_windows_records:
        pd.DataFrame(sig_windows_records).to_csv(
            os.path.join(out_dir, 'Decoding_v2_Significance_Windows.csv'), index=False)

    log.info('All figures written to %s', out_dir)


if __name__ == '__main__':
    main()
