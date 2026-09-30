"""Recognition accuracy of HTRN, BLT-VS and CORnet variants under occlusion (Supp. Fig. 5).

Four-way category accuracy (chance = 25%) of the linear readout on each model's top-area
features at 0%, 60% and 80% occlusion (8-fold CV, 20 repeats; bands are SEM). Dashed lines and
bands show human accuracy without and with a mask (Rajaei et al., 2019).

Eight series in three families:
    HTRN    FF = stage 0, LR = stage 6 (all blocks), TD = stage 7 (plus top-down feedback)
    BLT-VS  Feedforward (B) / Recurrent (BL) / TopDownFeedback (BLT)
    CORnet  Z (feedforward) / RT (recurrent)
The top row shows one panel per family; the bottom panel overlays all series. Colour encodes the
family (one hue each) and lightness the recurrence tier; markers differ for every series. A series
whose results file is missing is skipped.

Inputs:  results_{resnet_dfm,blt,cornet}_stages_multi_occ_train/multistage_4way_multi_occ_train_results.csv
         (``run_stage_sweep.py``), data/Human_behavioral_performance.csv
Outputs: Figure_Performance_Comparison (PNG/SVG/PDF) and Performance_Comparison_Data.csv in
         results_performance_comparison/

Usage:
    python scripts/07_figures/plot_model_performance.py
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from htrn import figstyle as fs, setup_logging
from htrn.config import DATA_ROOT, HUMAN_ACCURACY_CSV
from htrn.figstyle import INK, MID, HAIRLINE, GROUND

log = logging.getLogger(__name__)
fs.setup_rcparams()

# --- Data sources ---
RESNET_DFM_RESULTS = os.path.join(DATA_ROOT, 'results_resnet_dfm_stages_multi_occ_train',
                                  'multistage_4way_multi_occ_train_results.csv')
CORNET_RESULTS = os.path.join(DATA_ROOT, 'results_cornet_stages_multi_occ_train',
                              'multistage_4way_multi_occ_train_results.csv')
BLT_RESULTS = os.path.join(DATA_ROOT, 'results_blt_stages_multi_occ_train',
                           'multistage_4way_multi_occ_train_results.csv')
HUMAN_BEHAVIORAL_CSV = str(HUMAN_ACCURACY_CSV)

out_dir = os.path.join(DATA_ROOT, 'results_performance_comparison')

OCCLUSION_X = {'level_00': 0.0, 'level_60': 60.0, 'level_80': 80.0}
CHANCE_LEVEL_PCT = 25.0  # four categories

# Colours of the human reference lines
HUMAN_NOMASK_COLOR = '#9C7A1A'  # amber
HUMAN_MASK_COLOR = '#8B3A4A'    # wine

# (family, tier [0 = simplest .. 2 = fullest], results CSV, row filters, label, colour, marker).
# HTRN rows come from the stage sweep: FF = (LR, stage 0), LR = (LR, stage 6), TD = (TD, stage 7).
MODELS = [
    ('HTRN',   0, RESNET_DFM_RESULTS, {'model': 'DFM-ResNet', 'mechanism': 'LR', 'stage_idx': 0},
     'HTRN-FF (Feedforward)',      '#57A4FF', 'o'),
    ('HTRN',   1, RESNET_DFM_RESULTS, {'model': 'DFM-ResNet', 'mechanism': 'LR', 'stage_idx': 6},
     'HTRN-LR (Local-Recurrent)',  '#2A78D6', '^'),
    ('HTRN',   2, RESNET_DFM_RESULTS, {'model': 'DFM-ResNet', 'mechanism': 'TD', 'stage_idx': 7},
     'HTRN-TD (Top-Down)',         '#004DA7', 'D'),
    ('BLT-VS', 0, BLT_RESULTS, {'model': 'BLT-VS', 'stage': 'Feedforward'},
     'BLT-FF (Bottom-Up)',         '#FF9563', 's'),
    ('BLT-VS', 1, BLT_RESULTS, {'model': 'BLT-VS', 'stage': 'Recurrent'},
     'BLT-LR (Lateral-Recurrent)', '#EB6834', 'v'),
    ('BLT-VS', 2, BLT_RESULTS, {'model': 'BLT-VS', 'stage': 'TopDownFeedback'},
     'BLT-TD (Top-Down)',          '#BA3A00', 'P'),
    ('CORnet', 0, CORNET_RESULTS, {'model': 'CORNet-Z', 'stage': 'Feedforward'},
     'CORnet-Z (Feedforward)',     '#43C992', 'X'),
    ('CORnet', 1, CORNET_RESULTS, {'model': 'CORNet-RT', 'stage': 'Recurrent'},
     'CORnet-RT (Local-Recurrent)', '#1BAF7A', '*'),
]

FAMILY_ORDER = ['HTRN', 'BLT-VS', 'CORnet']


def load_series(csv_path, filters):
    """Mean and SEM accuracy (%) per occlusion level for the rows matching `filters`, or None."""
    if not os.path.exists(csv_path):
        log.warning('%s not found; skipping (run run_stage_sweep.py)', csv_path)
        return None
    df = pd.read_csv(csv_path)
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= (df[col].astype(str) == str(val))
    sub = df.loc[mask]
    if sub.empty:
        log.warning('No rows for filters=%r in %s', filters, csv_path)
        return None

    agg = sub.groupby('occlusion')['test_acc'].agg(['mean', 'sem'])
    xs, means, sems = [], [], []
    for occ_name, x in OCCLUSION_X.items():
        if occ_name in agg.index:
            xs.append(x)
            means.append(float(agg.loc[occ_name, 'mean']) * 100.0)
            sems.append(float(agg.loc[occ_name, 'sem']) * 100.0)
    if len(xs) < 2:
        log.warning('Fewer than 2 occlusion levels for filters=%r in %s', filters, csv_path)
        return None

    order = np.argsort(xs)
    return np.array(xs)[order], np.array(means)[order], np.array(sems)[order]


def load_human_behavioral(csv_path, group):
    """Human accuracy of one group ('No mask' or 'Mask') as (occlusion %, accuracy %, reported error)."""
    if not os.path.exists(csv_path):
        log.warning('%s not found; no human overlay', csv_path)
        return None
    df = pd.read_csv(csv_path)
    sub = df.loc[df['group'] == group].sort_values('occlusion_pct')
    if sub.empty:
        log.warning('No %r rows in %s', group, csv_path)
        return None
    return (sub['occlusion_pct'].to_numpy(dtype=float), sub['accuracy_pct'].to_numpy(dtype=float),
            sub['error_half_width'].to_numpy(dtype=float))


def draw_band(ax, xs, lo, hi, color, fill_alpha=0.15, edge_alpha=0.9, edge_width=0.9, zorder=2):
    """Shaded band with a thin opaque border (separate face and edge alpha)."""
    ax.fill_between(xs, lo, hi, facecolor=to_rgba(color, fill_alpha),
                    edgecolor=to_rgba(color, edge_alpha), linewidth=edge_width, zorder=zorder)


def draw_series(ax, s, markersize=6.5, linewidth=2.0):
    """Mean accuracy line with an SEM band for one model."""
    xs, means, sems, color = s['xs'], s['means'], s['sems'], s['color']
    draw_band(ax, xs, means - sems, means + sems, color, zorder=2)
    ax.plot(xs, means, color=color, linewidth=linewidth, solid_capstyle='round', zorder=4,
           marker=s['marker'], markersize=markersize, markeredgecolor=GROUND,
           markeredgewidth=1.0, label=s['legend_label'])


def draw_human_series(ax, xs, means, err, color, marker, dashes, label=None, markersize=6.0,
                      linewidth=2.0, fill_alpha=0.15, line_alpha=1.0):
    """Dashed human-accuracy line with its reported error band."""
    draw_band(ax, xs, means - err, means + err, color, fill_alpha=fill_alpha, zorder=3)
    ax.plot(xs, means, color=color, linewidth=linewidth, linestyle='--', dashes=dashes,
           solid_capstyle='round', zorder=5, marker=marker, markersize=markersize,
           markerfacecolor=GROUND, markeredgecolor=color, markeredgewidth=1.3,
           alpha=line_alpha, label=label)


def style_panel(ax):
    """Hairline spines, muted ticks, occlusion-level x ticks and a light y grid."""
    ax.set_facecolor(GROUND)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(HAIRLINE)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(labelsize=9.5, colors=MID, length=3.5, width=1.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(MID)
    ax.set_xticks([0, 60, 80])
    ax.set_xticklabels(['0%', '60%', '80%'])
    ax.grid(axis='y', color=HAIRLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    """Draw the performance figure and save its data."""
    setup_logging()
    os.makedirs(out_dir, exist_ok=True)
    series = []
    for family, tier, csv_path, filters, tag, color, marker in MODELS:
        data = load_series(csv_path, filters)
        if data is None:
            continue
        xs, means, sems = data
        series.append(dict(family=family, tier=tier, tag=tag, color=color, marker=marker,
                           xs=xs, means=means, sems=sems, legend_label=tag))

    if not series:
        log.error('No performance data found; run scripts/05_model_evaluation/run_stage_sweep.py first')
        return

    # Both human groups are reference lines only (the model stimuli have no mask split)
    HUMAN_GROUPS = [
        ('No mask', HUMAN_NOMASK_COLOR, 'o', (4, 2), 'Human (no mask)'),
        ('Mask',    HUMAN_MASK_COLOR,   's', (2, 2), 'Human (mask)'),
    ]
    humans = {}
    for group, color, marker, dashes, label in HUMAN_GROUPS:
        data = load_human_behavioral(HUMAN_BEHAVIORAL_CSV, group)
        if data is None:
            continue
        h_xs, h_means, h_err = data
        order = np.argsort(h_xs)
        humans[group] = dict(xs=h_xs[order], means=h_means[order], err=h_err[order],
                             color=color, marker=marker, dashes=dashes, label=label)

    all_means = np.concatenate([s['means'] for s in series]
                               + [h['means'] - h['err'] for h in humans.values()])
    all_sems = np.concatenate([s['sems'] for s in series]
                              + [np.zeros_like(h['means']) for h in humans.values()])
    y_lo = max(0.0, np.floor((all_means - all_sems).min() / 5.0) * 5.0 - 2.0)
    y_hi = 101.5

    FIGSIZE = (11.5, 9.0)
    fig = plt.figure(figsize=FIGSIZE, dpi=130, facecolor=GROUND)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35], hspace=0.34, wspace=0.10,
                         left=0.075, right=0.975, top=0.94, bottom=0.07)

    by_family = {}
    for s in series:
        by_family.setdefault(s['family'], []).append(s)

    top_axes = []
    for col, family in enumerate(FAMILY_ORDER):
        ax = fig.add_subplot(gs[0, col])
        top_axes.append(ax)
        members = sorted(by_family.get(family, []), key=lambda m: m['tier'])
        for m in members:
            draw_series(ax, m, markersize=5.5, linewidth=1.9)
        for h in humans.values():
            draw_human_series(ax, h['xs'], h['means'], h['err'], h['color'], h['marker'],
                              h['dashes'], markersize=4.5, linewidth=1.4, fill_alpha=0.10,
                              line_alpha=0.85)

        ax.set_ylim(y_lo, y_hi)
        style_panel(ax)
        ax.set_title(family, fontsize=11.5, color=INK, fontweight='bold', pad=8)
        if col == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=10, color=INK, labelpad=6)
        else:
            ax.tick_params(labelleft=False)
        if members:
            ax.legend(loc='lower left', fontsize=8.7, frameon=False, handlelength=1.6,
                     labelcolor=MID, borderaxespad=0.2)

    # Dedicated legend column, so the long entries cannot be clipped by the figure margin
    bottom_gs = gs[1, :].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.85], wspace=0.10)
    ax_all = fig.add_subplot(bottom_gs[0, :3])
    ax_legend = fig.add_subplot(bottom_gs[0, 3])
    ax_legend.axis('off')

    for s in sorted(series, key=lambda s: (FAMILY_ORDER.index(s['family']), s['tier'])):
        draw_series(ax_all, dict(s, legend_label=s['tag']),
                   markersize=7.0, linewidth=2.3)

    for h in humans.values():
        draw_human_series(ax_all, h['xs'], h['means'], h['err'], h['color'], h['marker'],
                          h['dashes'], label=h['label'], markersize=6.0, linewidth=2.0,
                          fill_alpha=0.15)

    ax_all.axhline(CHANCE_LEVEL_PCT, color=HAIRLINE, linewidth=1.2, linestyle=':', zorder=1)

    ax_all.set_ylim(y_lo, y_hi)
    style_panel(ax_all)
    ax_all.set_xlabel('Occlusion level', fontsize=11.5, color=INK, labelpad=8,
                      fontweight='bold')
    ax_all.set_ylabel('Accuracy (%)', fontsize=11.5, color=INK, labelpad=8,
                      fontweight='bold')

    handles, labels = ax_all.get_legend_handles_labels()
    ax_legend.legend(handles, labels, loc='center left', fontsize=9.3, frameon=False,
                     labelcolor=MID, handlelength=1.7, borderaxespad=0, title='Model',
                     title_fontsize=9.8)
    ax_legend.get_legend().get_title().set_color(INK)
    ax_legend.get_legend().get_title().set_fontweight('bold')
    ax_legend.text(0.0, -0.03, 'Chance level = 25%',
                  transform=ax_legend.transAxes, fontsize=8.3, color=MID, fontstyle='italic',
                  va='top', ha='left')

    missing = [f"{m[0]} {m[4]}" for m in MODELS
              if not any(s['family'] == m[0] and s['tag'] == m[4] for s in series)]
    if missing:
        log.warning('Skipped models without data: %s', ', '.join(missing))

    paths = fs.save_figure(fig, out_dir, 'Figure_Performance_Comparison')
    plt.close(fig)
    log.info('Saved %s', paths[0])

    rows = []
    for s in series:
        for x, mean, sem in zip(s['xs'], s['means'], s['sems']):
            rows.append({'Family': s['family'], 'Tag': s['tag'], 'Occlusion_pct': x,
                        'Mean_Accuracy_pct': mean, 'SEM_pct': sem})
    for group, h in humans.items():
        for x, mean, err in zip(h['xs'], h['means'], h['err']):
            rows.append({'Family': 'Human', 'Tag': f'Behavioral ({group.lower()})',
                        'Occlusion_pct': x, 'Mean_Accuracy_pct': mean, 'SEM_pct': err})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, 'Performance_Comparison_Data.csv'),
                              index=False)


if __name__ == '__main__':
    main()
