"""Representational Granger causality between ROIs (Fig. 3B and Supp. Figs. 2-4).

Grand-average GC time courses (0-300 ms) for both directions between each pair of ROIs, after
subtracting each subject's pre-stimulus baseline.

Two kinds of figure are produced:

* Three per-pair figures (Supp. Figs. 2-4; ``Figure_RDM_GC_Direction_<pair>``): rows are the
  bottom-up trace, the top-down trace and their difference (top-down minus bottom-up, no mask);
  columns are occlusion levels. Each trace panel overlays no-mask and mask GC with a rail of
  mask-vs-no-mask differences below it.
* One overview (Fig. 3B; ``Figure_RDM_GC_Direction_All``): rows are occlusion levels plus the
  occlusion effect (60% minus 0%), columns are ROI pairs; both directions are overlaid, with a rail
  comparing them.

Statistics (cluster-permutation tests): GC against baseline is one-tailed (a log variance ratio is
only interpretable above baseline); mask vs no-mask, direction vs direction and occlusion effects
are two-tailed. A difference is reported only where at least one of the two traces is itself
significant. Mask-80 does not exist, so only 0% and 60% occlusion are shown.

Inputs:  Megocclusion/derivatives/RDM_GC/ (``compute_rdm_granger.py``)
Outputs: figures (PNG/SVG/PDF) and latency / significance CSVs in ``RDM_GC/plots_RDM_GC_Direction_v2/``

Usage:
    python scripts/07_figures/plot_granger_causality.py
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
    filter_diff_clusters,
    style_axes, draw_legend_row, draw_ledger, draw_rail_single, draw_diff_panel,
    draw_cascade, save_figure, report_page_fit,
)

log = logging.getLogger(__name__)
fs.setup_rcparams()

derivatives_dir = str(DERIVATIVES_DIR)
gc_dir = os.path.join(derivatives_dir, 'RDM_GC')
out_dir = os.path.join(gc_dir, 'plots_RDM_GC_Direction_v2')

sub_ids = SUBJECT_IDS
occlusion_levels = ['0', '60']

# One colour per ordered ROI pair (direction of information flow). Markers repeat the
# distinction: bottom-up 'o', top-down 's', mask 'D'.
FF_V13_LOC   = '#EB5A47'   # V1-3   -> LOC      coral
FB_LOC_V13   = '#27187E'   # LOC    -> V1-3     deep indigo
FF_LOC_ITPHC = '#E5417B'   # LOC    -> IT-PHC   raspberry
FB_ITPHC_LOC = '#F2A900'   # IT-PHC -> LOC      marigold
FF_V13_ITPHC = '#1E88E5'   # V1-3   -> IT-PHC   azure
FB_ITPHC_V13 = '#1DB954'   # IT-PHC -> V1-3     emerald

# (block id, pair title, bottom-up (source, target, colour), top-down (source, target, colour))
BLOCKS = [
    ('V13_LOC',   'V1-3 ↔ LOC',
     ('V1-3', 'LOC', FF_V13_LOC),      ('LOC', 'V1-3', FB_LOC_V13)),
    ('LOC_ITPHC', 'LOC ↔ IT-PHC',
     ('LOC', 'IT-PHC', FF_LOC_ITPHC),  ('IT-PHC', 'LOC', FB_ITPHC_LOC)),
    ('V13_ITPHC', 'V1-3 ↔ IT-PHC',
     ('V1-3', 'IT-PHC', FF_V13_ITPHC), ('IT-PHC', 'V1-3', FB_ITPHC_V13)),
]

FF_MARKER, FB_MARKER, MASK_MARKER = 'o', 's', 'D'
TRACE_LW = fs.pt(1.45)
MASK_LW = fs.pt(1.25)
DIFF_PANEL_RATIO = 0.46    # height of the difference panel relative to a trace panel
RAIL_RATIO = 0.16          # one lane per rail

# The GC traces start at stimulus onset, so the plotted window is also the tested window.
# GC values are left unscaled; matplotlib shows the exponent as offset text.
TMIN, TMAX = 0.0, 0.300
TICK_STEP = 0.05
ALPHA = 0.05
SIGMA_SMOOTH = 2.0
N_PERMUTATIONS = 'all'
GC_SCALE = 1.0

# Legends span the full canvas width so the overview's six entries fit on one line
LEGEND_X0 = 0.006
LEGEND_X_MAX = 0.994

# --- Layout ---
# Per pair: two trace rows plus a shorter difference row, one column per occlusion level
PAIR_FIGSIZE = (fs.FIG_W, 9.60)
PAIR_GRID = dict(hspace=0.13, wspace=0.13, left=0.098, right=0.988, top=0.912, bottom=0.058)
PAIR_LEGEND_Y = 0.982
PAIR_COLTITLE_Y = 0.940

# Overview: two occlusion-level rows plus the occlusion-effect row, one column per ROI pair
OVERVIEW_FIGSIZE = (fs.FIG_W, 9.70)
OVERVIEW_GRID = dict(hspace=0.15, wspace=0.15, left=0.098, right=0.988, top=0.87, bottom=0.055)
OVERVIEW_LEGEND_Y = 0.983
OVERVIEW_LEGEND_DY = 0.024   # line spacing if the legend wraps
OVERVIEW_COLTITLE_Y = 0.918

_CLUSTER_CACHE = {}


def style_time_axis(ax, ticklabels=False, label=False):
    """Time axis with this figure's limits and tick spacing."""
    fs.style_time_axis(ax, TMIN, TMAX, TICK_STEP, ticklabels=ticklabels, label=label)


def style_gc_yaxis(ax):
    """Scientific-notation y-axis; the offset text is moved into the axes by ``place_gc_offset_inside``."""
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))


def place_gc_offset_inside(fig, axes, anchor='left'):
    """Move each axis's scientific-notation offset text just inside its top corner.

    Call after all y-limits are final. Trace panels anchor top-left; difference panels (whose
    ledger occupies the top-left) anchor top-right.
    """
    fig.canvas.draw()
    x, ha = (0.014, 'left') if anchor == 'left' else (0.986, 'right')
    for ax in axes:
        offset = ax.yaxis.get_offset_text()
        text = offset.get_text()
        if not text:
            continue
        offset.set_visible(False)
        ax.text(x, 0.965, text, transform=ax.transAxes, ha=ha, va='top',
                fontsize=fs.FS_TICK, color=MID, zorder=15)


def run_clusters_cached(cache_key, X, t_threshold, stat_mask, tail):
    """``run_clusters`` with memoisation (per-pair and overview figures repeat identical tests)."""
    if cache_key not in _CLUSTER_CACHE:
        _CLUSTER_CACHE[cache_key] = run_clusters(X, t_threshold, stat_mask, ALPHA,
                                                 N_PERMUTATIONS, tail=tail)
    return _CLUSTER_CACHE[cache_key]


# --- Loading ---

def load_gc_time_vector():
    """Time vector of the saved GC traces (0-300 ms, 1 ms step by default)."""
    files = sorted(glob.glob(os.path.join(gc_dir, 'sub-*', '*_gc_*.npz')))
    if files:
        return np.load(files[0])['times']
    return np.linspace(0.0, 0.300, 301)


_GC_CACHE = {}


def load_gc(source, target, level, mask_cond):
    """Baseline-corrected GC (subjects, times) of one direction and condition, or None if any
    subject is missing. The baseline is stored as a scalar, so correction is a subtraction."""
    key = (source, target, level, mask_cond)
    if key in _GC_CACHE:
        return _GC_CACHE[key]
    cond = level if mask_cond == 'nomask' else f'{mask_cond}-{level}'
    group_data = []
    for sub in sub_ids:
        fpath = os.path.join(gc_dir, f'sub-{sub:02d}',
                             f'sub-{sub:02d}_occlusion-{cond}_gc_{source}-to-{target}.npz')
        if not os.path.exists(fpath):
            continue
        with np.load(fpath) as d:
            group_data.append((d['gc_raw'] - d['baseline']) * GC_SCALE)
    result = np.array(group_data) if len(group_data) == len(sub_ids) else None
    _GC_CACHE[key] = result
    return result


# --- Panels ---

def draw_trace_panel(fig, ax, rail, source, target, dir_color, dir_marker, level, pair_title,
                     plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two,
                     loso_peaks_records, vs_base_records, sig_windows_records,
                     loso_onsets_records, show_sd=True):
    """One trace cell: no-mask (direction colour) and mask (neutral) GC for one direction and
    occlusion level, plus the mask-vs-no-mask rail below.

    Returns (y_lo, y_hi, no-mask traces, indices significant against baseline) for the caller's
    shared limits and difference row.
    """
    direction = f'{source}-to-{target}'
    conds = [('nomask', 'No mask', dir_color, dir_marker, '-'),
             ('mask', 'Mask', MASK_COLOR, MASK_MARKER, '-')]

    traces = {}
    vs_base = {}
    ledger = []
    y_hi, y_lo = -np.inf, np.inf

    for cond_key, cond_label, color, marker, ls in conds:
        X_full = load_gc(source, target, level, cond_key)
        if X_full is None:
            continue
        X = X_full[:, plot_mask]
        traces[cond_key] = X

        peaks_loso, _ = get_loso_latencies(X, plot_times, peak_mask, SIGMA_SMOOTH, baseline_val=0)
        peak_mean = np.mean(peaks_loso)
        peak_sd = np.std(peaks_loso, ddof=1)
        for i_sub, sub in enumerate(sub_ids):
            loso_peaks_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Condition': cond_key, 'Value': peaks_loso[i_sub] * 1000})

        # One-tailed against baseline (tail=1, paired with t_one)
        onsets_loso, _, _ = get_jackknife_onset_latencies(
            X, plot_times, stat_mask, alpha=ALPHA, n_permutations=N_PERMUTATIONS, tail=1)
        for i_sub, sub in enumerate(sub_ids):
            loso_onsets_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Condition': cond_key, 'Value': onsets_loso[i_sub] * 1000})

        mean_smooth = gaussian_filter1d(np.mean(X, axis=0), sigma=SIGMA_SMOOTH)
        sem_smooth = gaussian_filter1d(sem(X, axis=0), sigma=SIGMA_SMOOTH)
        y_hi = max(y_hi, np.max(mean_smooth + sem_smooth))
        y_lo = min(y_lo, np.min(mean_smooth - sem_smooth), 0.0)

        z = 2 if cond_key == 'nomask' else 3
        lw = TRACE_LW if cond_key == 'nomask' else MASK_LW
        ax.fill_between(plot_times, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        color=color, alpha=0.14, linewidth=0, zorder=z)
        ax.plot(plot_times, mean_smooth, color=color, linewidth=lw, linestyle=ls,
                solid_capstyle='round', zorder=6 + z)

        sig_idx = []
        for idx, p in run_clusters_cached(
                ('base', direction, level, cond_key), X, t_one, stat_mask, 1):
            ax.plot(plot_times[idx], mean_smooth[idx], color=color, linewidth=fs.LW_TRACE_SIG,
                    linestyle=ls, solid_capstyle='round', zorder=6 + z)
            sig_idx.append(idx)
            vs_base_records.append({
                'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Condition': cond_key, 'Start_Time_ms': plot_times[idx[0]] * 1000,
                'End_Time_ms': plot_times[idx[-1]] * 1000, 'P_Value': p,
                'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})
        vs_base[cond_key] = set(np.concatenate(sig_idx).tolist()) if sig_idx else set()

        # Report the peak only if it lies in a significant cluster
        t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
        if t_idx in vs_base[cond_key]:
            ax.plot(peak_mean, mean_smooth[t_idx], marker=marker, color=color,
                    markersize=fs.MS_PEAK if marker != 'D' else fs.MS_PEAK_SM, zorder=20,
                    markeredgecolor='white', markeredgewidth=fs.pt(0.70), linestyle='none')
            ledger.append((cond_label, color, marker, peak_mean * 1000, peak_sd * 1000))

    # No-mask vs mask, two-tailed, gated on either side's own significance
    lane_clusters = []
    if 'nomask' in traces and 'mask' in traces:
        data_diff = traces['nomask'] - traces['mask']
        mean_diff = data_diff.mean(axis=0)
        raw = run_clusters_cached(('mask', direction, level), data_diff, t_two, stat_mask, 0)
        lane_clusters = filter_diff_clusters(raw, vs_base['nomask'], vs_base['mask'], mean_diff)
        for idx, p, win_is_a in lane_clusters:
            sig_windows_records.append({
                'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Comparison': 'No mask vs Mask',
                'Winner': 'No mask' if win_is_a else 'Mask',
                'Loser': 'Mask' if win_is_a else 'No mask',
                'Start_Time_ms': plot_times[idx[0]] * 1000,
                'End_Time_ms': plot_times[idx[-1]] * 1000, 'P_Value': p,
                'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})

    ax.axhline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, zorder=1)
    style_axes(ax)
    style_time_axis(ax)
    style_gc_yaxis(ax)
    draw_ledger(fig, ax, ledger, show_sd=show_sd)

    draw_rail_single(rail, [(idx, w) for idx, _p, w in lane_clusters],
                     dir_color, MASK_COLOR, plot_times, TMIN, TMAX)

    span = max(y_hi - y_lo, 1e-6)
    return (y_lo - 0.06 * span, y_hi + 0.06 * span,
            traces.get('nomask'), vs_base.get('nomask', set()))


def _prepare():
    """Time axis, masks and t thresholds (one-tailed for baseline tests, two-tailed for differences)."""
    full_times = load_gc_time_vector()
    plot_mask = (full_times >= TMIN) & (full_times <= TMAX)
    plot_times = full_times[plot_mask]
    n_subjects = len(sub_ids)
    t_one = stats.t.ppf(1 - ALPHA, n_subjects - 1)        # tail=1
    t_two = stats.t.ppf(1 - ALPHA / 2, n_subjects - 1)    # tail=0
    return plot_times, plot_mask, plot_times >= TMIN, plot_times >= TMIN, t_one, t_two


def _write_csvs(fig_id, loso_peaks_records, vs_base_records, sig_windows_records,
                loso_onsets_records=None):
    """Write the latency and significance CSVs of one figure."""
    save_loso_csv(loso_peaks_records,
                  os.path.join(out_dir, f'RDM_GC_v2_LOSO_Peak_Latencies_{fig_id}.csv'))
    if loso_onsets_records is not None:
        save_loso_csv(loso_onsets_records,
                      os.path.join(out_dir, f'RDM_GC_v2_LOSO_Onset_Latencies_{fig_id}.csv'))
    if vs_base_records:
        pd.DataFrame(vs_base_records).to_csv(
            os.path.join(out_dir, f'RDM_GC_v2_VsBaseline_Clusters_{fig_id}.csv'), index=False)
    if sig_windows_records:
        pd.DataFrame(sig_windows_records).to_csv(
            os.path.join(out_dir, f'RDM_GC_v2_Significance_Windows_{fig_id}.csv'), index=False)


def build_pair_figure(block):
    """One per-pair figure: bottom-up and top-down trace rows plus their difference row."""
    block_id, pair_title, (ff_source, ff_target, ff_color), (fb_source, fb_target, fb_color) = block
    log.info("Figure: %s (%s)", block_id, pair_title)

    plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two = _prepare()
    loso_peaks_records, vs_base_records, sig_windows_records = [], [], []
    loso_onsets_records = []

    fig = plt.figure(figsize=PAIR_FIGSIZE, dpi=110, facecolor=GROUND)
    outer = fig.add_gridspec(3, len(occlusion_levels),
                             height_ratios=[1.0 + RAIL_RATIO, 1.0 + RAIL_RATIO,
                                            DIFF_PANEL_RATIO + RAIL_RATIO],
                             **PAIR_GRID)

    legend_items = [
        (f'{ff_source} → {ff_target}', ff_color, FF_MARKER, fs.MS_LEGEND, '-', TRACE_LW),
        (f'{fb_source} → {fb_target}', fb_color, FB_MARKER, fs.MS_LEGEND, '-', TRACE_LW),
        ('Mask', MASK_COLOR, MASK_MARKER, fs.MS_LEGEND_SM, '-', MASK_LW),
    ]
    draw_legend_row(fig, legend_items, PAIR_LEGEND_Y, LEGEND_X0, x_max=LEGEND_X_MAX)

    col0_axes, row_labels, row_colors, col_title_x = [], [], [], []
    trace_axes = []
    nomask_traces = {}

    for d_idx, (source, target, color, marker, dir_key) in enumerate(
            [(ff_source, ff_target, ff_color, FF_MARKER, 'ff'),
             (fb_source, fb_target, fb_color, FB_MARKER, 'fb')]):
        row_labels.append(f'{source} → {target}')
        row_colors.append(color)

        for col_idx, level in enumerate(occlusion_levels):
            cell = outer[d_idx, col_idx].subgridspec(
                2, 1, height_ratios=[1.0, RAIL_RATIO], hspace=0.05)
            ax = fig.add_subplot(cell[0])
            rail = fig.add_subplot(cell[1], sharex=ax)
            if col_idx == 0:
                col0_axes.append(ax)
            if d_idx == 0:
                col_title_x.append(ax.get_position().x0)

            y_lo, y_hi, X_nomask, vs_base_idx = draw_trace_panel(
                fig, ax, rail, source, target, color, marker, level, pair_title,
                plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two,
                loso_peaks_records, vs_base_records, sig_windows_records,
                loso_onsets_records)
            trace_axes.append((ax, y_lo, y_hi))
            nomask_traces[(dir_key, level)] = (X_nomask, vs_base_idx)

            style_time_axis(rail)
            if col_idx == 0:
                ax.set_ylabel('GC', fontsize=fs.FS_AXLABEL, color=INK,
                              labelpad=fs.pt(1.8))

    # Difference row: top-down minus bottom-up, no mask
    row_labels.append('Δ  top-down − bottom-up')
    row_colors.append(INK)
    diff_axes = []
    for col_idx, level in enumerate(occlusion_levels):
        cell = outer[2, col_idx].subgridspec(
            2, 1, height_ratios=[DIFF_PANEL_RATIO, RAIL_RATIO], hspace=0.09)
        ax_diff = fig.add_subplot(cell[0])
        rail = fig.add_subplot(cell[1], sharex=ax_diff)
        if col_idx == 0:
            col0_axes.append(ax_diff)
        style_gc_yaxis(ax_diff)

        X_ff, base_ff = nomask_traces[('ff', level)]
        X_fb, base_fb = nomask_traces[('fb', level)]
        lane_clusters = []
        if X_ff is not None and X_fb is not None:
            data_diff = X_fb - X_ff          # positive = top-down ahead
            mean_diff = data_diff.mean(axis=0)
            raw = run_clusters_cached(('dir', block_id, level), data_diff, t_two, stat_mask, 0)
            lane_clusters = filter_diff_clusters(raw, base_fb, base_ff, mean_diff)
            fb_name = f'{fb_source}-to-{fb_target}'
            ff_name = f'{ff_source}-to-{ff_target}'
            for idx, p, win_is_a in lane_clusters:
                sig_windows_records.append({
                    'Pair': pair_title, 'Direction': f'{fb_name} vs {ff_name}',
                    'Level': level, 'Comparison': 'Top-down vs bottom-up (no mask)',
                    'Winner': fb_name if win_is_a else ff_name,
                    'Loser': ff_name if win_is_a else fb_name,
                    'Start_Time_ms': plot_times[idx[0]] * 1000,
                    'End_Time_ms': plot_times[idx[-1]] * 1000, 'P_Value': p,
                    'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})

            twin_max = draw_diff_panel(
                fig, ax_diff, plot_times, data_diff, lane_clusters, fb_color, ff_color,
                peak_mask, SIGMA_SMOOTH, loso_peaks_records,
                {'Pair': pair_title, 'Direction': f'{fb_name} vs {ff_name}', 'Level': level,
                 'Condition': 'Difference (FB − FF)'},
                sub_ids, 'Δ peak', TMIN, TMAX, TICK_STEP, onset_guide=False,
                loso_onsets_records=loso_onsets_records, stat_mask=stat_mask,
                alpha=ALPHA, n_permutations=N_PERMUTATIONS, tail=0)
            diff_axes.append((ax_diff, twin_max))

        draw_rail_single(rail, [(idx, w) for idx, _p, w in lane_clusters],
                         fb_color, ff_color, plot_times, TMIN, TMAX)
        style_time_axis(rail, ticklabels=True, label=True)
        if col_idx == 0:
            ax_diff.set_ylabel('Δ GC (top-down − bottom-up)', fontsize=fs.FS_AXLABEL_SM,
                               color=INK, labelpad=fs.pt(2.6))

    # Share limits across the four trace panels so directions and levels compare by eye
    if trace_axes:
        block_lo = min(lo for (_a, lo, _hi) in trace_axes)
        block_hi = max(hi for (_a, _lo, hi) in trace_axes)
        for ax, _lo, _hi in trace_axes:
            ax.set_ylim(block_lo, block_hi)
    if diff_axes:
        shared_twin = max(tm for (_a, tm) in diff_axes) * 1.40
        for ax_diff, _tm in diff_axes:
            ax_diff.set_ylim(-shared_twin, shared_twin)

    for x0, level in zip(col_title_x, occlusion_levels):
        fig.text(x0, PAIR_COLTITLE_Y, f'{level}% occlusion', fontsize=fs.FS_COLTITLE,
                 fontweight='bold', color=INK, ha='left', va='bottom')

    draw_cascade(fig, col0_axes, row_labels, row_colors)

    place_gc_offset_inside(fig, [a for a, _lo, _hi in trace_axes], anchor='left')
    place_gc_offset_inside(fig, [a for a, _tm in diff_axes], anchor='right')

    _write_csvs(block_id, loso_peaks_records, vs_base_records, sig_windows_records,
               loso_onsets_records)
    paths = save_figure(fig, out_dir, f'Figure_RDM_GC_Direction_{block_id}')
    report_page_fit(PAIR_FIGSIZE, f'Figure_RDM_GC_Direction_{block_id}')
    plt.close(fig)
    log.info("Saved %s", paths[0])


def draw_overview_panel(fig, ax, rail, pair_title, level, dir_a, dir_b,
                        plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two,
                        loso_peaks_records, vs_base_records, sig_windows_records,
                        loso_onsets_records):
    """One overview cell: both directions of one ROI pair (no mask) overlaid, with a rail
    comparing the two directions.

    Returns the panel's padded (y_lo, y_hi) for column-wise limit sharing.
    """
    dirs = [dir_a, dir_b]
    traces, vs_base, ledger = {}, {}, []
    y_hi, y_lo = -np.inf, np.inf

    for source, target, color, marker in dirs:
        direction = f'{source}-to-{target}'
        X_full = load_gc(source, target, level, 'nomask')
        if X_full is None:
            continue
        X = X_full[:, plot_mask]
        traces[direction] = X

        peaks_loso, _ = get_loso_latencies(X, plot_times, peak_mask, SIGMA_SMOOTH, baseline_val=0)
        peak_mean = np.mean(peaks_loso)
        peak_sd = np.std(peaks_loso, ddof=1)
        for i_sub, sub in enumerate(sub_ids):
            loso_peaks_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Condition': 'nomask', 'Value': peaks_loso[i_sub] * 1000})

        onsets_loso, _, _ = get_jackknife_onset_latencies(
            X, plot_times, stat_mask, alpha=ALPHA, n_permutations=N_PERMUTATIONS, tail=1)
        for i_sub, sub in enumerate(sub_ids):
            loso_onsets_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': level,
                'Condition': 'nomask', 'Value': onsets_loso[i_sub] * 1000})

        mean_smooth = gaussian_filter1d(np.mean(X, axis=0), sigma=SIGMA_SMOOTH)
        sem_smooth = gaussian_filter1d(sem(X, axis=0), sigma=SIGMA_SMOOTH)
        y_hi = max(y_hi, np.max(mean_smooth + sem_smooth))
        y_lo = min(y_lo, np.min(mean_smooth - sem_smooth), 0.0)

        ax.fill_between(plot_times, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(plot_times, mean_smooth, color=color, linewidth=TRACE_LW,
                solid_capstyle='round', zorder=8)

        # One-tailed against baseline; separate cache key from the per-pair figures
        sig_idx = []
        for idx, p in run_clusters_cached(('base_ov', direction, level), X, t_one, stat_mask, 1):
            ax.plot(plot_times[idx], mean_smooth[idx], color=color, linewidth=fs.LW_TRACE_SIG,
                    solid_capstyle='round', zorder=8)
            sig_idx.append(idx)
            vs_base_records.append({
                'Pair': pair_title, 'Direction': direction, 'Level': level, 'Condition': 'nomask',
                'Start_Time_ms': plot_times[idx[0]] * 1000, 'End_Time_ms': plot_times[idx[-1]] * 1000,
                'P_Value': p, 'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})
        vs_base[direction] = set(np.concatenate(sig_idx).tolist()) if sig_idx else set()

        t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
        if t_idx in vs_base[direction]:
            ax.plot(peak_mean, mean_smooth[t_idx], marker=marker, color=color,
                    markersize=fs.MS_PEAK, zorder=20,
                    markeredgecolor='white', markeredgewidth=fs.pt(0.70), linestyle='none')
            ledger.append((f'{source}→{target}', color, marker, peak_mean * 1000, peak_sd * 1000))

    # Bottom-up vs top-down, two-tailed, gated on either direction's own significance
    src_a, tgt_a, color_a, _m_a = dir_a
    src_b, tgt_b, color_b, _m_b = dir_b
    dir_a_name, dir_b_name = f'{src_a}-to-{tgt_a}', f'{src_b}-to-{tgt_b}'
    lane_clusters = []
    if dir_a_name in traces and dir_b_name in traces:
        data_diff = traces[dir_a_name] - traces[dir_b_name]
        mean_diff = data_diff.mean(axis=0)
        raw = run_clusters_cached(('dir_ov', pair_title, level), data_diff, t_two, stat_mask, 0)
        lane_clusters = filter_diff_clusters(raw, vs_base[dir_a_name], vs_base[dir_b_name], mean_diff)
        for idx, p, win_is_a in lane_clusters:
            sig_windows_records.append({
                'Pair': pair_title, 'Direction': f'{dir_a_name} vs {dir_b_name}',
                'Level': level, 'Comparison': 'Bottom-up vs top-down (no mask)',
                'Winner': dir_a_name if win_is_a else dir_b_name,
                'Loser': dir_b_name if win_is_a else dir_a_name,
                'Start_Time_ms': plot_times[idx[0]] * 1000,
                'End_Time_ms': plot_times[idx[-1]] * 1000, 'P_Value': p,
                'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})

    ax.axhline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, zorder=1)
    style_axes(ax)
    style_time_axis(ax)
    style_gc_yaxis(ax)
    draw_ledger(fig, ax, ledger, show_sd=False)

    draw_rail_single(rail, [(idx, w) for idx, _p, w in lane_clusters],
                     color_a, color_b, plot_times, TMIN, TMAX)

    span = max(y_hi - y_lo, 1e-6)
    return y_lo - 0.06 * span, y_hi + 0.06 * span


def draw_overview_diff_panel(fig, ax, rail, pair_title, dir_a, dir_b,
                             plot_times, plot_mask, peak_mask, stat_mask, t_two,
                             loso_peaks_records, vs_base_records, sig_windows_records,
                             loso_onsets_records):
    """Third-row cell: the occlusion effect (60% minus 0%, no mask) for both directions.

    Each direction is tested against zero (two-tailed, as the effect need not be positive), and a
    rail compares the two directions. Returns the padded (y_lo, y_hi) for row-wise limit sharing.
    """
    dirs = [dir_a, dir_b]
    traces, vs_zero, ledger = {}, {}, []
    y_hi, y_lo = -np.inf, np.inf

    for source, target, color, marker in dirs:
        direction = f'{source}-to-{target}'
        X0_full = load_gc(source, target, '0', 'nomask')
        X60_full = load_gc(source, target, '60', 'nomask')
        if X0_full is None or X60_full is None:
            continue
        Xd = X60_full[:, plot_mask] - X0_full[:, plot_mask]
        traces[direction] = Xd

        # The effect can go either way, so the peak is the largest |value|
        peaks_loso, _ = get_loso_latencies(Xd, plot_times, peak_mask, SIGMA_SMOOTH, baseline_val=0,
                                           use_abs=True)
        peak_mean = np.mean(peaks_loso)
        peak_sd = np.std(peaks_loso, ddof=1)
        for i_sub, sub in enumerate(sub_ids):
            loso_peaks_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': '60-0',
                'Condition': 'occlusion_effect', 'Value': peaks_loso[i_sub] * 1000})

        onsets_loso, _, _ = get_jackknife_onset_latencies(
            Xd, plot_times, stat_mask, alpha=ALPHA, n_permutations=N_PERMUTATIONS, tail=0)
        for i_sub, sub in enumerate(sub_ids):
            loso_onsets_records.append({
                'Subject': sub, 'Pair': pair_title, 'Direction': direction, 'Level': '60-0',
                'Condition': 'occlusion_effect', 'Value': onsets_loso[i_sub] * 1000})

        mean_smooth = gaussian_filter1d(np.mean(Xd, axis=0), sigma=SIGMA_SMOOTH)
        sem_smooth = gaussian_filter1d(sem(Xd, axis=0), sigma=SIGMA_SMOOTH)
        y_hi = max(y_hi, np.max(mean_smooth + sem_smooth))
        y_lo = min(y_lo, np.min(mean_smooth - sem_smooth))

        ax.fill_between(plot_times, mean_smooth - sem_smooth, mean_smooth + sem_smooth,
                        color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(plot_times, mean_smooth, color=color, linewidth=TRACE_LW,
                solid_capstyle='round', zorder=8)

        sig_idx = []
        for idx, p in run_clusters_cached(('occl_base', direction), Xd, t_two, stat_mask, 0):
            ax.plot(plot_times[idx], mean_smooth[idx], color=color, linewidth=fs.LW_TRACE_SIG,
                    solid_capstyle='round', zorder=8)
            sig_idx.append(idx)
            vs_base_records.append({
                'Pair': pair_title, 'Direction': direction, 'Level': '60-0',
                'Condition': 'occlusion_effect',
                'Start_Time_ms': plot_times[idx[0]] * 1000, 'End_Time_ms': plot_times[idx[-1]] * 1000,
                'P_Value': p, 'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})
        vs_zero[direction] = set(np.concatenate(sig_idx).tolist()) if sig_idx else set()

        t_idx = int(np.argmin(np.abs(plot_times - peak_mean)))
        if t_idx in vs_zero[direction]:
            ax.plot(peak_mean, mean_smooth[t_idx], marker=marker, color=color,
                    markersize=fs.MS_PEAK_SM, zorder=20,
                    markeredgecolor='white', markeredgewidth=fs.pt(0.70), linestyle='none')
            ledger.append((f'{source}→{target}', color, marker, peak_mean * 1000, peak_sd * 1000))

    # Does occlusion move the two directions differently? Gated as above
    src_a, tgt_a, color_a, _m_a = dir_a
    src_b, tgt_b, color_b, _m_b = dir_b
    dir_a_name, dir_b_name = f'{src_a}-to-{tgt_a}', f'{src_b}-to-{tgt_b}'
    lane_clusters = []
    if dir_a_name in traces and dir_b_name in traces:
        data_diff = traces[dir_a_name] - traces[dir_b_name]
        mean_diff = data_diff.mean(axis=0)
        raw = run_clusters_cached(('occl_dir', pair_title), data_diff, t_two, stat_mask, 0)
        lane_clusters = filter_diff_clusters(raw, vs_zero[dir_a_name], vs_zero[dir_b_name], mean_diff)
        for idx, p, win_is_a in lane_clusters:
            sig_windows_records.append({
                'Pair': pair_title, 'Direction': f'{dir_a_name} vs {dir_b_name}',
                'Level': '60-0', 'Comparison': 'Occlusion effect: bottom-up vs top-down',
                'Winner': dir_a_name if win_is_a else dir_b_name,
                'Loser': dir_b_name if win_is_a else dir_a_name,
                'Start_Time_ms': plot_times[idx[0]] * 1000,
                'End_Time_ms': plot_times[idx[-1]] * 1000, 'P_Value': p,
                'Stat_Window': f'{int(TMIN * 1000)}-{int(TMAX * 1000)}ms'})

    # Zero means no occlusion effect, so the line is darker than in trace panels
    ax.axhline(0, color=MID, linewidth=fs.LW_GUIDE, zorder=1)
    style_axes(ax)
    style_time_axis(ax)
    style_gc_yaxis(ax)
    draw_ledger(fig, ax, ledger, show_sd=False)

    draw_rail_single(rail, [(idx, w) for idx, _p, w in lane_clusters],
                     color_a, color_b, plot_times, TMIN, TMAX)

    span = max(y_hi - y_lo, 1e-6)
    return y_lo - 0.08 * span, y_hi + 0.08 * span


def build_overview_figure(blocks):
    """Overview of all ROI pairs (occlusion-level rows plus the occlusion-effect row).

    The two raw-GC rows share y-limits down each column; the difference row shares its limits
    across columns.
    """
    log.info('Figure: overview')
    plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two = _prepare()
    loso_peaks_records, vs_base_records, sig_windows_records = [], [], []
    loso_onsets_records = []

    fig = plt.figure(figsize=OVERVIEW_FIGSIZE, dpi=110, facecolor=GROUND)
    outer = fig.add_gridspec(3, len(blocks),
                             height_ratios=[1.0 + RAIL_RATIO, 1.0 + RAIL_RATIO,
                                            DIFF_PANEL_RATIO + RAIL_RATIO],
                             **OVERVIEW_GRID)

    # One legend entry per direction (no mask entry: the overview has no mask condition)
    legend_items = []
    for _bid, _pt, (fsrc, ftgt, fcol), (bsrc, btgt, bcol) in blocks:
        legend_items.append((f'{fsrc} → {ftgt}', fcol, FF_MARKER, fs.MS_LEGEND, '-', TRACE_LW))
        legend_items.append((f'{bsrc} → {btgt}', bcol, FB_MARKER, fs.MS_LEGEND, '-', TRACE_LW))
    draw_legend_row(fig, legend_items, OVERVIEW_LEGEND_Y, LEGEND_X0,
                    x_max=LEGEND_X_MAX, dy=OVERVIEW_LEGEND_DY)

    col0_axes, col_title_x = [], []
    row_labels, row_colors = [], []
    columns_data = [[] for _ in blocks]

    for row_idx, level in enumerate(occlusion_levels):
        row_labels.append(f'{level}% occlusion')
        row_colors.append(INK)

        for col_idx, (block_id, pair_title, ff, fb) in enumerate(blocks):
            dir_a = (ff[0], ff[1], ff[2], FF_MARKER)
            dir_b = (fb[0], fb[1], fb[2], FB_MARKER)

            cell = outer[row_idx, col_idx].subgridspec(
                2, 1, height_ratios=[1.0, RAIL_RATIO], hspace=0.05)
            ax = fig.add_subplot(cell[0])
            rail = fig.add_subplot(cell[1], sharex=ax)
            if col_idx == 0:
                col0_axes.append(ax)
            if row_idx == 0:
                col_title_x.append(ax.get_position().x0)

            y_lo, y_hi = draw_overview_panel(
                fig, ax, rail, pair_title, level, dir_a, dir_b,
                plot_times, plot_mask, peak_mask, stat_mask, t_one, t_two,
                loso_peaks_records, vs_base_records, sig_windows_records,
                loso_onsets_records)
            columns_data[col_idx].append((ax, y_lo, y_hi))

            # Time labels are drawn on the bottom (difference) row only
            style_time_axis(rail, ticklabels=False, label=False)
            if col_idx == 0:
                ax.set_ylabel('GC', fontsize=fs.FS_AXLABEL, color=INK, labelpad=fs.pt(1.8))

    # Share y-limits down each column (0% and 60% occlusion of one pair)
    for entries in columns_data:
        col_lo = min(lo for (_a, lo, _hi) in entries)
        col_hi = max(hi for (_a, _lo, hi) in entries)
        for ax, _lo, _hi in entries:
            ax.set_ylim(col_lo, col_hi)

    # Third row: occlusion effect (60% - 0%), y-limits shared across the ROI pairs
    row_labels.append('Δ  60% − 0% occlusion')
    row_colors.append(INK)
    diff_entries = []

    for col_idx, (block_id, pair_title, ff, fb) in enumerate(blocks):
        dir_a = (ff[0], ff[1], ff[2], FF_MARKER)
        dir_b = (fb[0], fb[1], fb[2], FB_MARKER)

        cell = outer[2, col_idx].subgridspec(
            2, 1, height_ratios=[DIFF_PANEL_RATIO, RAIL_RATIO], hspace=0.09)
        ax = fig.add_subplot(cell[0])
        rail = fig.add_subplot(cell[1], sharex=ax)
        if col_idx == 0:
            col0_axes.append(ax)

        y_lo, y_hi = draw_overview_diff_panel(
            fig, ax, rail, pair_title, dir_a, dir_b,
            plot_times, plot_mask, peak_mask, stat_mask, t_two,
            loso_peaks_records, vs_base_records, sig_windows_records,
            loso_onsets_records)
        diff_entries.append((ax, y_lo, y_hi))

        style_time_axis(rail, ticklabels=True, label=True)
        if col_idx == 0:
            ax.set_ylabel('Δ GC', fontsize=fs.FS_AXLABEL_SM, color=INK, labelpad=fs.pt(2.6))

    diff_lo = min(lo for (_a, lo, _hi) in diff_entries)
    diff_hi = max(hi for (_a, _lo, hi) in diff_entries)
    for ax, _lo, _hi in diff_entries:
        ax.set_ylim(diff_lo, diff_hi)

    for x0, (_block_id, pair_title, _ff, _fb) in zip(col_title_x, blocks):
        fig.text(x0, OVERVIEW_COLTITLE_Y, pair_title, fontsize=fs.FS_COLTITLE,
                 fontweight='bold', color=INK, ha='left', va='bottom')

    draw_cascade(fig, col0_axes, row_labels, row_colors)

    place_gc_offset_inside(fig, [a for col in columns_data for a, _lo, _hi in col]
                           + [a for a, _lo, _hi in diff_entries])

    _write_csvs('All', loso_peaks_records, vs_base_records, sig_windows_records,
               loso_onsets_records)
    paths = save_figure(fig, out_dir, 'Figure_RDM_GC_Direction_All')
    report_page_fit(OVERVIEW_FIGSIZE, 'Figure_RDM_GC_Direction_All')
    plt.close(fig)
    log.info("Saved %s", paths[0])


def main():
    """Draw the three per-pair figures and the overview."""
    setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    for block in BLOCKS:
        build_pair_figure(block)
    build_overview_figure(BLOCKS)
    log.info('All figures written to %s', out_dir)


if __name__ == '__main__':
    main()
