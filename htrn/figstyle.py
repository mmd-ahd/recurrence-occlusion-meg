"""Shared figure style and helpers for the time-series figures (decoding, RSA, Granger causality).

Every figure uses the same canvas width (``FIG_W``) and expresses sizes as printed points through
``pt()`` (``SCALE`` converts to the 2x canvas), so text and line weights print at the same size in
all figures. ``save_figure`` writes with ``bbox_inches=None`` to keep the saved width equal to the
declared one; cropping to content would change the print scale per figure.

Analysis settings (subjects, time windows, alpha, smoothing, paths) live in each figure script.
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from scipy import stats
from scipy.stats import sem
from scipy.ndimage import gaussian_filter1d
from mne.stats import spatio_temporal_cluster_1samp_test

log = logging.getLogger(__name__)


# --- Scale ---

SCALE = 2.0            # canvas is 2x final print size
FIG_W = 16                        # canvas width (in)
PAGE_MAX_RATIO = 247.0 / 183.0    # tallest height/width that fits a 183 x 247 mm page


def pt(x):
    """Convert a size in printed points to the value to pass to matplotlib on the scaled canvas."""
    return x * SCALE


# --- Type sizes (the argument to pt() is what lands on paper) ---
FS_TICK = pt(6.5)
FS_AXLABEL = pt(7.5)
FS_AXLABEL_SM = pt(6.5)     # secondary y-labels, e.g. the diverging difference panel
FS_LEGEND = pt(8.0)
FS_COLTITLE = pt(9.0)
FS_COLTITLE_SM = pt(7.5)    # second-tier column label under a spanning group title
FS_ROWLABEL = pt(8.0)
FS_LEDGER = pt(6.0)
FS_RAIL = pt(6.0)

# --- Line weights ---
LW_SPINE = pt(0.60)
LW_GUIDE = pt(0.60)         # zero line and dotted stimulus-onset guide
LW_TRACE_SIG = pt(2.70)     # significant stretch redrawn over its own trace
LW_RAIL = pt(4.50)
LW_DIFF = pt(1.00)
LW_DIFF_SIG = pt(2.20)
LW_CASCADE = pt(0.90)
TICK_LEN = pt(2.00)
TICK_W = pt(0.60)

# --- Marker sizes ---
MS_PEAK = pt(4.50)
MS_PEAK_SM = pt(4.00)       # 'D' and 'X' read larger than 'o'/'s' at equal markersize
MS_LEDGER = pt(3.60)
MS_LEDGER_SM = pt(3.20)
MS_LEGEND = pt(5.00)
MS_LEGEND_SM = pt(4.50)
MS_CASCADE = pt(5.00)


# --- Palette ("Luminous Jewel on white") ---

INK = '#101114'
MID = '#5F656D'
FAINT = '#9AA0A8'
HAIRLINE = '#E4E7EB'
TRACK = '#F2F4F7'
GROUND = '#FFFFFF'

# Near-neutral colour of the second state (masked) of a series
MASK_COLOR = '#4A4F57'
MASK_DASH = (0, (4.5, 2.0))

# One colour per ROI (used in the decoding figure); model mechanisms and Granger-causality
# directions have their own palettes in the figure scripts.
ROI_COLORS = {
    'V1-3':   '#e41a1c',
    'LOC':    '#377eb8',
    'IT-PHC': '#5ab4ac',
}


def setup_rcparams():
    """Set fonts and keep text editable (not outlined) in SVG and PDF exports."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


# --- Statistics helpers ---

def get_loso_latencies(data_matrix, times, peak_mask, smoothing_sigma, baseline_val=0,
                       use_abs=False):
    """Leave-one-subject-out peak and half-height onset latencies of the group mean.

    Args:
        data_matrix: (subjects, times) array.
        times: Time vector matching the columns of ``data_matrix``.
        peak_mask: Boolean mask of the window searched for the peak.
        smoothing_sigma: Gaussian smoothing (samples) applied to each subsample mean.
        baseline_val: Reference value for the half-height onset.
        use_abs: Search the peak of the absolute value (for signed differences).

    Returns:
        (peaks, onsets): one latency per left-out subject.
    """
    n_subs = data_matrix.shape[0]
    loso_peaks = []
    loso_onsets = []
    for i in range(n_subs):
        train_data = np.delete(data_matrix, i, axis=0)
        mean_train = np.mean(train_data, axis=0)
        mean_train_smooth = gaussian_filter1d(mean_train, sigma=smoothing_sigma)

        window_times = times[peak_mask]
        window_vals = mean_train_smooth[peak_mask]

        if use_abs:
            peak_idx = np.argmax(np.abs(window_vals))
        else:
            peak_idx = np.argmax(window_vals)

        peak_time = window_times[peak_idx]
        peak_val = window_vals[peak_idx]
        loso_peaks.append(peak_time)

        target_val = baseline_val + 0.5 * (peak_val - baseline_val)
        pre_peak_vals = window_vals[:peak_idx + 1]
        pre_peak_times = window_times[:peak_idx + 1]

        if (peak_val >= baseline_val) if not use_abs else (peak_val >= 0):
            crossings = np.where(pre_peak_vals <= target_val)[0]
        else:
            crossings = np.where(pre_peak_vals >= target_val)[0]

        if len(crossings) > 0:
            onset_idx = crossings[-1]
            if onset_idx < len(pre_peak_vals) - 1:
                t1, t2 = pre_peak_times[onset_idx], pre_peak_times[onset_idx + 1]
                v1, v2 = pre_peak_vals[onset_idx], pre_peak_vals[onset_idx + 1]
                if v2 != v1:
                    ratio = (target_val - v1) / (v2 - v1)
                    onset_time = t1 + ratio * (t2 - t1)
                else:
                    onset_time = t1
            else:
                onset_time = pre_peak_times[onset_idx]
        else:
            onset_time = pre_peak_times[0]

        loso_onsets.append(onset_time)

    return np.array(loso_peaks), np.array(loso_onsets)


def get_jackknife_onset_latencies(data_matrix, times, test_mask, alpha=0.05,
                                   n_permutations='all', tail=0):
    """Jackknife onset latency from cluster-permutation tests.

    For each leave-one-subject-out subsample, the first time point of its earliest significant
    cluster is taken as that subsample's onset.

    Returns:
        (onsets, mean, sd) over subsamples. If any subsample has no significant cluster its onset
        is NaN and mean/sd are NaN (onset not estimable).
    """
    n_subs = data_matrix.shape[0]
    df = (n_subs - 1) - 1
    t_threshold = stats.t.ppf(1 - alpha / (2 if tail == 0 else 1), df)

    onsets = np.full(n_subs, np.nan)
    for i in range(n_subs):
        sub_data = np.delete(data_matrix, i, axis=0)
        X = sub_data[:, :, np.newaxis]
        sig_clusters = run_clusters(X, t_threshold, test_mask, alpha, n_permutations, tail)
        if not sig_clusters:
            continue
        onsets[i] = min(times[idx].min() for idx, p in sig_clusters)

    if np.any(np.isnan(onsets)):
        return onsets, np.nan, np.nan

    jackknife_mean = onsets.mean()
    jackknife_sd = onsets.std(ddof=1)
    return onsets, jackknife_mean, jackknife_sd


def save_loso_csv(records, filepath):
    """Save per-subject latency records with appended Mean and SD rows."""
    if not records:
        return
    df = pd.DataFrame(records)
    group_cols = [c for c in df.columns if c not in ['Subject', 'Value']]

    summary_mean = df.groupby(group_cols)['Value'].mean().reset_index()
    summary_mean['Subject'] = 'Mean'

    summary_sd = df.groupby(group_cols)['Value'].std(ddof=1).reset_index()
    summary_sd['Subject'] = 'SD'

    df_final = pd.concat([df, summary_mean, summary_sd], ignore_index=True)
    cols = ['Subject'] + group_cols + ['Value']
    df_final[cols].to_csv(filepath, index=False)


def run_clusters(X, t_threshold, stat_mask, alpha, n_permutations='all', tail=0):
    """One-sample cluster-permutation test over the masked time window.

    `t_threshold` and `tail` must be chosen together (two-tailed unless testing an increase only).

    Returns:
        [(indices into the full time vector, p_value), ...] for clusters with p < alpha.
    """
    offset = int(np.argmax(stat_mask))
    X_stat = X[:, stat_mask][:, :, np.newaxis]
    _, clusters, cluster_p, _ = spatio_temporal_cluster_1samp_test(
        X_stat, threshold=t_threshold, n_permutations=n_permutations,
        tail=tail, n_jobs=-1, verbose=False, out_type='indices'
    )
    return [(c[0] + offset, p) for c, p in zip(clusters, cluster_p) if p < alpha]


def filter_diff_clusters(diff_clusters, vs_zero_idx_a, vs_zero_idx_b, mean_diff):
    """Keep difference clusters that overlap a time point where either trace differs from zero.

    Returns [(indices, p, first_series_wins), ...].
    """
    vs_zero_idx = vs_zero_idx_a | vs_zero_idx_b
    kept = []
    for idx, p in diff_clusters:
        if vs_zero_idx.isdisjoint(idx.tolist()):
            continue
        win_is_a = bool(mean_diff[idx].mean() > 0)
        kept.append((idx, p, win_is_a))
    return kept


# --- Axis styling ---

def style_axes(ax):
    """Minimal spines and muted ticks."""
    ax.set_facecolor(GROUND)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(HAIRLINE)
        ax.spines[side].set_linewidth(LW_SPINE)
    ax.tick_params(labelsize=FS_TICK, colors=MID, length=TICK_LEN, width=TICK_W)
    for lbl in ax.get_yticklabels() + ax.get_xticklabels():
        lbl.set_color(MID)


def style_time_axis(ax, tmin, tmax, tick_step, ticklabels=False, label=False):
    """Time axis in ms. Ticks are drawn on every panel; labels only where requested (bottom rows)."""
    ticks = np.arange(tmin, tmax + 1e-9, tick_step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t * 1000:.0f}' for t in ticks])
    ax.set_xlim(tmin, tmax)
    ax.tick_params(labelbottom=ticklabels)
    if label:
        ax.set_xlabel('Time (ms)', fontsize=FS_AXLABEL, color=INK, labelpad=pt(1.8))


def _measure(fig, artist):
    """Rendered extent of `artist` in figure fractions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    return artist.get_window_extent(renderer=renderer).transformed(fig.transFigure.inverted())


def _text_width_fig(fig, s, fontsize, weight='bold'):
    """Rendered width of `s` in figure fractions."""
    probe = fig.text(0, -1, s, fontsize=fontsize, fontweight=weight)
    w = _measure(fig, probe).width
    probe.remove()
    return w


# --- Legend ---

def draw_legend_row(fig, items, y, x0, x_max=None, gap=0.020, dy=0.016):
    """Legend row with item widths measured from the rendered text; wraps at `x_max`.

    items: [(label, color, marker, markersize, linestyle, linewidth), ...]; use linestyle=None for
    a marker-only entry.
    """
    x, y_cur = x0, y
    for label, color, marker, markersize, ls, lw in items:
        draw_line = ls is not None and lw
        width = _text_width_fig(fig, label, FS_LEGEND)
        item_width = width + (0.036 if draw_line else 0.018)
        if x_max is not None and x > x0 and x + item_width > x_max:
            x, y_cur = x0, y_cur - dy

        if draw_line:
            fig.add_artist(Line2D([x, x + 0.027], [y_cur, y_cur], color=color,
                                  linewidth=lw, linestyle=ls))
            marker_x, text_x = x + 0.0135, x + 0.037
        else:
            marker_x, text_x = x, x + 0.018
        fig.add_artist(Line2D([marker_x], [y_cur], marker=marker, color=color, linestyle='none',
                              markersize=markersize, markeredgecolor=GROUND,
                              markeredgewidth=pt(0.65), zorder=10))
        fig.text(text_x, y_cur, label, fontsize=FS_LEGEND, color=INK,
                 ha='left', va='center', fontweight='bold')
        x = text_x + width + gap
    return y_cur


# --- Peak-latency ledger ---

LEDGER_Y = 1.038                # axes fraction, just above the top spine
LEDGER_MARKER_PAD_IN = 0.105     # marker centre to text start (in)
LEDGER_GAP_IN = 0.150            # end of one entry to the next marker (in)


def draw_ledger(fig, ax, entries, y=LEDGER_Y, x0=0.008, line_dy=0.085, show_sd=True):
    """Peak-latency ledger above a panel: one "<label> mean ± SD ms" entry per series.

    Entries are spaced from measured text widths and wrap to a further line at the panel edge.
    entries: [(label, color, marker, mean_ms, sd_ms), ...]; `show_sd=False` drops the SD for narrow
    panels.
    """
    if not entries:
        return

    def _fmt(s, mu, sd):
        return f'{s} {mu:.0f} ± {sd:.0f} ms' if show_sd else f'{s} {mu:.0f} ms'

    widths = [_text_width_fig(fig, _fmt(s, mu, sd), FS_LEDGER)
              for s, _c, _m, mu, sd in entries]
    ax_w = _measure(fig, ax).width
    widths_ax = [w / ax_w for w in widths]
    # Padding and gap are given in inches so they do not shrink with the panel width
    ax_w_in = ax_w * fig.get_size_inches()[0]
    marker_pad = LEDGER_MARKER_PAD_IN / ax_w_in
    gap = LEDGER_GAP_IN / ax_w_in

    x, y_cur = x0, y
    for (short, color, marker, mu_ms, sd_ms), w in zip(entries, widths_ax):
        item_w = marker_pad + w
        if x > x0 and x + item_w > 1.0:
            x, y_cur = x0, y_cur + line_dy
        ax.plot([x], [y_cur], marker=marker, color=color,
                markersize=MS_LEDGER if marker not in ('D', 'X') else MS_LEDGER_SM,
                transform=ax.transAxes, clip_on=False, linestyle='none', zorder=25)
        ax.text(x + marker_pad, y_cur, _fmt(short, mu_ms, sd_ms),
                transform=ax.transAxes, fontsize=FS_LEDGER, color=color,
                fontweight='bold', va='center', ha='left', zorder=25)
        x += item_w + gap


# --- Significance rails ---

def _style_rail(rail, n_lanes):
    """Style a significance rail: only a bottom hairline, no y ticks."""
    rail.set_facecolor(GROUND)
    if n_lanes > 1:
        rail.set_ylim(-0.62, n_lanes - 0.38)
    else:
        rail.set_ylim(-0.5, 0.5)
    for side in ('top', 'right', 'left'):
        rail.spines[side].set_visible(False)
    rail.spines['bottom'].set_color(HAIRLINE)
    rail.spines['bottom'].set_linewidth(LW_SPINE)
    rail.set_yticks([])
    rail.tick_params(labelsize=FS_TICK, colors=MID, length=TICK_LEN, width=TICK_W)


def draw_rail_single(rail, lane_clusters, color_a, color_b, times, tmin, tmax):
    """Single-comparison rail with winner-coloured significant clusters; 'n.s.' if none."""
    _style_rail(rail, 1)
    track_start = max(0.0, tmin)
    rail.plot([track_start, tmax], [0, 0], color=TRACK, lw=LW_RAIL,
              solid_capstyle='butt', zorder=1)
    if lane_clusters:
        for idx, win_is_a in lane_clusters:
            rail.plot([times[idx[0]], times[idx[-1]]], [0, 0],
                      color=color_a if win_is_a else color_b, lw=LW_RAIL,
                      solid_capstyle='round', zorder=5)
    else:
        rail.text((track_start + tmax) / 2, 0, 'n.s.', color=FAINT, fontsize=FS_RAIL,
                  fontweight='bold', va='center', ha='center', zorder=5)


def draw_rail_multi(fig, rail, lanes, times, tmin, tmax):
    """Rail with one lane per pairwise comparison; significant clusters take the winner's colour.

    Each lane head shows filled (winner) and hollow (loser) chips plus a "winner > loser" label, or
    hollow chips and "a ~ b" when nothing is significant. `lanes` is a list of
    (name_a, color_a, name_b, color_b, clusters) with clusters as (indices, p, a_wins).
    """
    n_lanes = len(lanes)
    _style_rail(rail, n_lanes)

    head = blended_transform_factory(rail.transAxes, rail.transData)
    ax_w = _measure(fig, rail).width
    zero_frac = (0.0 - tmin) / (tmax - tmin)    # axes fraction where the track begins
    chip_gap = pt(4.6) / 72.0 / (ax_w * FIG_W)  # chip spacing in axes fractions

    for lane_idx, (sa, ca, sb, cb, clusters) in enumerate(lanes):
        y = n_lanes - 1 - lane_idx
        rail.plot([0.0, tmax], [y, y], color=TRACK, lw=LW_RAIL,
                  solid_capstyle='butt', zorder=1)

        if clusters:
            biggest = max(clusters, key=lambda c: len(c[0]))
            win_is_a = biggest[2]
            win_s, win_c = (sa, ca) if win_is_a else (sb, cb)
            lose_s, lose_c = (sb, cb) if win_is_a else (sa, ca)
            label, label_color = f'{win_s} > {lose_s}', win_c
        else:
            win_c = lose_c = None
            label, label_color = f'{sa} ~ {sb}', FAINT

        text_x = zero_frac - 0.014 - _text_width_fig(fig, label, FS_RAIL) / ax_w
        chip_b_x = text_x - chip_gap
        chip_a_x = chip_b_x - chip_gap

        if clusters:
            rail.scatter([chip_a_x], [y], s=pt(4.1) ** 2, facecolor=win_c, edgecolor='none',
                         transform=head, zorder=4, clip_on=False)
            rail.scatter([chip_b_x], [y], s=pt(4.1) ** 2, facecolor=GROUND, edgecolor=lose_c,
                         linewidth=pt(0.95), transform=head, zorder=4, clip_on=False)
        else:
            for xo, c in ((chip_a_x, ca), (chip_b_x, cb)):
                rail.scatter([xo], [y], s=pt(4.1) ** 2, facecolor=GROUND, edgecolor=c,
                             linewidth=pt(0.95), transform=head, zorder=4, clip_on=False)
        rail.text(text_x, y, label, color=label_color, fontsize=FS_RAIL,
                  fontweight='bold', va='center', ha='left', transform=head,
                  zorder=4, clip_on=False)

        for idx, _p, win_is_a in clusters:
            rail.plot([times[idx[0]], times[idx[-1]]], [y, y],
                      color=ca if win_is_a else cb, lw=LW_RAIL,
                      solid_capstyle='round', zorder=5)


# --- Diverging difference panel ---

DIFF_MARKER = 'X'


def draw_diff_panel(fig, ax_diff, plot_times, data_diff, lane_clusters, color_pos, color_neg,
                    peak_mask, sigma_smooth, loso_peaks_records, record_base, sub_ids,
                    ledger_label, tmin, tmax, tick_step, scale=1.0, onset_guide=True,
                    ledger_y=0.88, loso_onsets_records=None, stat_mask=None,
                    alpha=0.05, n_permutations='all', tail=0):
    """Diverging A-minus-B panel: filled in `color_pos` above zero (A ahead), `color_neg` below.

    Significant windows (`lane_clusters`, computed by the caller) get a stronger fill and line. The
    peak-latency records (LOSO) are appended to `loso_peaks_records` and, if `stat_mask` is given,
    the jackknife onsets to `loso_onsets_records`.

    Returns:
        The panel's maximum |value|, so callers can share a symmetric y-limit across a row.
    """
    diff_mean = gaussian_filter1d(data_diff.mean(axis=0), sigma=sigma_smooth) * scale
    diff_sem = gaussian_filter1d(sem(data_diff, axis=0), sigma=sigma_smooth) * scale

    ax_diff.fill_between(plot_times, diff_mean - diff_sem, diff_mean + diff_sem,
                         color=MID, alpha=0.10, linewidth=0, zorder=2)
    ax_diff.fill_between(plot_times, 0, diff_mean, where=(diff_mean >= 0), interpolate=True,
                         color=color_pos, alpha=0.18, linewidth=0, zorder=3)
    ax_diff.fill_between(plot_times, 0, diff_mean, where=(diff_mean < 0), interpolate=True,
                         color=color_neg, alpha=0.18, linewidth=0, zorder=3)

    pos_line = np.where(diff_mean >= 0, diff_mean, np.nan)
    neg_line = np.where(diff_mean < 0, diff_mean, np.nan)
    ax_diff.plot(plot_times, pos_line, color=color_pos, linewidth=LW_DIFF,
                 solid_capstyle='round', zorder=4)
    ax_diff.plot(plot_times, neg_line, color=color_neg, linewidth=LW_DIFF,
                 solid_capstyle='round', zorder=4)

    for idx, _p, win_is_a in lane_clusters:
        c = color_pos if win_is_a else color_neg
        ax_diff.fill_between(plot_times[idx], 0, diff_mean[idx], color=c, alpha=0.42,
                             linewidth=0, zorder=5, interpolate=True)
        ax_diff.plot(plot_times[idx], diff_mean[idx], color=c, linewidth=LW_DIFF_SIG,
                     solid_capstyle='round', zorder=6)

    twin_abs_max = float(np.max(np.abs(diff_mean) + diff_sem))

    # The difference can peak in either direction, so the peak search uses |value|
    diff_ledger = []
    diff_peaks_loso, _ = get_loso_latencies(data_diff, plot_times, peak_mask, sigma_smooth,
                                            baseline_val=0, use_abs=True)
    diff_peak_mean = np.mean(diff_peaks_loso)
    diff_peak_sd = np.std(diff_peaks_loso, ddof=1)
    for i_sub, sub in enumerate(sub_ids):
        rec = dict(record_base)
        rec['Subject'] = sub
        rec['Value'] = diff_peaks_loso[i_sub] * 1000
        loso_peaks_records.append(rec)

    if loso_onsets_records is not None and stat_mask is not None:
        diff_onsets_loso, _, _ = get_jackknife_onset_latencies(
            data_diff, plot_times, stat_mask, alpha=alpha,
            n_permutations=n_permutations, tail=tail)
        for i_sub, sub in enumerate(sub_ids):
            rec = dict(record_base)
            rec['Subject'] = sub
            rec['Value'] = diff_onsets_loso[i_sub] * 1000
            loso_onsets_records.append(rec)

    sig_flat = set()
    for idx, _p, _w in lane_clusters:
        sig_flat.update(idx.tolist())
    t_idx = int(np.argmin(np.abs(plot_times - diff_peak_mean)))
    if lane_clusters and t_idx in sig_flat:
        peak_val = diff_mean[t_idx]
        peak_color = color_pos if peak_val >= 0 else color_neg
        ax_diff.plot(diff_peak_mean, peak_val, marker=DIFF_MARKER, color=peak_color,
                     markersize=MS_PEAK_SM, zorder=20, markeredgecolor='white',
                     markeredgewidth=pt(0.70), linestyle='none')
        diff_ledger.append((ledger_label, peak_color, DIFF_MARKER,
                            diff_peak_mean * 1000, diff_peak_sd * 1000))

    # The zero line marks the crossover between the two series, so it is darker than in trace panels
    ax_diff.axhline(0, color=MID, linewidth=LW_GUIDE, zorder=1)
    if onset_guide:
        ax_diff.axvline(0, color=HAIRLINE, linewidth=LW_GUIDE, linestyle=(0, (1, 2.4)), zorder=1)
    style_axes(ax_diff)
    style_time_axis(ax_diff, tmin, tmax, tick_step)

    # Ledger inside the panel (the gap above is too narrow); the top corner is empty by symmetry
    draw_ledger(fig, ax_diff, diff_ledger, y=ledger_y)

    return twin_abs_max


# --- Side cascade rail ---

CASCADE_RAIL_X = 0.030
CASCADE_LABEL_X = 0.0125


def draw_cascade(fig, axes, labels, colors, markers=None,
                 rail_x=CASCADE_RAIL_X, label_x=CASCADE_LABEL_X, fontsize=FS_ROWLABEL):
    """Vertical rail down the left edge with a labelled dot at each row's vertical centre."""
    centers = [(a.get_position().y0 + a.get_position().y1) / 2 for a in axes]
    if len(centers) > 1:
        fig.add_artist(Line2D([rail_x, rail_x], [centers[-1], centers[0]],
                              color=HAIRLINE, linewidth=LW_CASCADE, zorder=0))

    # Shrink rotated labels that would be taller than their row (down to a legible minimum)
    fig_h = fig.get_size_inches()[1]
    if len(centers) > 1:
        allowance = abs(centers[0] - centers[1]) * 0.94
    else:
        allowance = 0.94
    sizes = []
    for label in labels:
        w_in = _text_width_fig(fig, label, fontsize) * fig.get_size_inches()[0]
        needed = w_in / fig_h  # rotated text: its length becomes figure height
        sizes.append(fontsize if needed <= allowance
                     else max(pt(5.5), fontsize * allowance / needed))

    markers = markers or ['o'] * len(centers)
    for c, label, color, marker, size in zip(centers, labels, colors, markers, sizes):
        fig.add_artist(Line2D([rail_x], [c], marker=marker, markersize=MS_CASCADE, color=color,
                              markeredgecolor=GROUND, markeredgewidth=pt(1.15), linestyle='none'))
        fig.text(label_x, c, label, rotation=90, fontsize=size,
                 fontweight='bold', color=color, ha='center', va='center')
    return centers


# --- Export ---

def save_figure(fig, out_dir, stem, dpi=600):
    """Save PNG, SVG and PDF at exactly the figure's own size (no tight cropping)."""
    paths = []
    for ext in ('png', 'svg', 'pdf'):
        p = os.path.join(out_dir, f'{stem}.{ext}')
        fig.savefig(p, dpi=dpi if ext == 'png' else None, facecolor=GROUND, bbox_inches=None)
        paths.append(p)
    return paths


def report_page_fit(figsize, name):
    """Log whether a figure's height/width fits the 183 x 247 mm page limit."""
    w, h = figsize
    ratio = h / w
    flag = 'OK' if ratio <= PAGE_MAX_RATIO else 'TOO TALL'
    log.info('%s: %.2f x %.2f in, H/W = %.3f (limit %.3f) -> %s', name, w, h, ratio, PAGE_MAX_RATIO, flag)
