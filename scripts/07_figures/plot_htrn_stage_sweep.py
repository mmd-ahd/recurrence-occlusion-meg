"""HTRN readout-stage sweep: recognition accuracy as recurrence depth grows (Fig. 4C).

Four-way category accuracy of the linear readout (``run_stage_sweep.py --model htrn``) at every
readout stage, for 0%, 60% and 80% occlusion. LR (local recurrence only) covers stages 0-6 and TD
(local recurrence plus top-down feedback) stages 0-7. Stage 0 is the feedforward readout (HTRN-FF)
and is identical for LR and TD, so it is marked once with an open ring. Curves show the mean over
CV folds and repeats with SEM bands; dashed bands show human accuracy (no mask / mask, from
Rajaei et al., 2019) with its reported error. No hypothesis tests are drawn.

Inputs:  results_resnet_dfm_stages_multi_occ_train/multistage_4way_multi_occ_train_results.csv
         data/Human_behavioral_performance.csv
Outputs: Figure_HTRN_StageSweep (PNG/SVG/PDF) and StageSweep_Data.csv in
         results_resnet_dfm_stages_multi_occ_train/plots_StageSweep_v2/

Usage:
    python scripts/07_figures/plot_htrn_stage_sweep.py
"""

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba

from htrn import figstyle as fs, setup_logging
from htrn.config import DATA_ROOT, HUMAN_ACCURACY_CSV
from htrn.figstyle import (INK, HAIRLINE, GROUND, draw_legend_row, save_figure, report_page_fit,
                           style_axes)

log = logging.getLogger(__name__)
fs.setup_rcparams()

RESULTS_DIR = DATA_ROOT / 'results_resnet_dfm_stages_multi_occ_train'
RESULTS_CSV = os.path.join(RESULTS_DIR, 'multistage_4way_multi_occ_train_results.csv')
MODEL_NAME = 'DFM-ResNet'   # the fine-tuned HTRN
out_dir = os.path.join(RESULTS_DIR, 'plots_StageSweep_v2')

# (occlusion key in the results, panel title, occlusion %)
OCCLUSION_LEVELS = [
    ('level_00', '0% occlusion', 0.0),
    ('level_60', '60% occlusion', 60.0),
    ('level_80', '80% occlusion', 80.0),
]

HUMAN_CSV = str(HUMAN_ACCURACY_CSV)
HUMAN_NOMASK_COLOR = '#9C7A1A'
HUMAN_MASK_COLOR = '#8B3A4A'
# (group, colour, marker, dash pattern, legend label)
HUMAN_GROUPS = [
    ('No mask', HUMAN_NOMASK_COLOR, 'o', (4, 2), 'Human (no mask)'),
    ('Mask',    HUMAN_MASK_COLOR,   's', (2, 2), 'Human (mask)'),
]

FF_COLOR, FF_MARKER, FF_LABEL = '#F4256D', 'o', 'Feedforward (FF)'

# (mechanism id in the results, label, short label, colour, marker, linewidth)
MECHANISMS = [
    ('LR', 'Local-Recurrent (LR)', 'LR', '#3A0CA3', 's', fs.pt(1.50)),
    ('TD', 'Top-Down (TD)',        'TD', '#00A5A8', 'D', fs.pt(1.50)),
]

# The five legend entries wrap onto two rows, hence the larger line spacing
FIGSIZE = (fs.FIG_W, 5.9)
GRID = dict(wspace=0.09, left=0.052, right=0.985, top=0.70, bottom=0.135)
LEGEND_Y = 0.975
LEGEND_DY = 0.062
COLTITLE_Y = 0.775


def draw_band(ax, xs, lo, hi, color, fill_alpha=0.16, edge_alpha=0.9, zorder=2):
    """Shaded band with a thin opaque border (separate face and edge alpha)."""
    ax.fill_between(xs, lo, hi, facecolor=to_rgba(color, fill_alpha),
                    edgecolor=to_rgba(color, edge_alpha), linewidth=fs.pt(0.55), zorder=zorder)


def load_stage_sweep(csv_path, model_name):
    """Mean and SEM of test accuracy per mechanism, stage and occlusion level."""
    df = pd.read_csv(csv_path)
    df = df.loc[df['model'] == model_name]
    return df.groupby(['mechanism', 'stage_idx', 'occlusion'])['test_acc'] \
        .agg(['mean', 'sem']).reset_index()


def load_human_behavioral(csv_path):
    """Human accuracy table, or None if the file is missing."""
    if not os.path.exists(csv_path):
        log.warning('%s not found; no human overlay', csv_path)
        return None
    return pd.read_csv(csv_path)


def main():
    """Draw the stage-sweep figure and save its data."""
    setup_logging()
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(RESULTS_CSV):
        log.error('%s not found; run scripts/05_model_evaluation/run_stage_sweep.py --model htrn first',
                  RESULTS_CSV)
        return

    agg = load_stage_sweep(RESULTS_CSV, MODEL_NAME)
    if agg.empty:
        log.error('No rows for model %r in %s', MODEL_NAME, RESULTS_CSV)
        return

    human_df = load_human_behavioral(HUMAN_CSV)

    series_by_occ = {occ_key: {} for occ_key, _, _ in OCCLUSION_LEVELS}
    ff_origin_by_occ = {}
    human_by_occ = {occ_key: {} for occ_key, _, _ in OCCLUSION_LEVELS}
    all_means, all_sems = [], []
    export_rows = []

    for occ_key, occ_label, occ_pct in OCCLUSION_LEVELS:
        for mech_id, disp_label, short, color, marker, lw in MECHANISMS:
            sub = agg.loc[(agg['mechanism'] == mech_id) & (agg['occlusion'] == occ_key)] \
                .sort_values('stage_idx')
            if sub.empty:
                continue
            xs = sub['stage_idx'].to_numpy(dtype=float)
            means = sub['mean'].to_numpy(dtype=float) * 100.0
            sems = sub['sem'].to_numpy(dtype=float) * 100.0
            series_by_occ[occ_key][mech_id] = dict(xs=xs, means=means, sems=sems)
            all_means.append(means)
            all_sems.append(sems)
            for x, m, s in zip(xs, means, sems):
                export_rows.append({'Mechanism': short, 'Stage': int(x),
                                    'Occlusion_pct': occ_pct, 'Mean_Accuracy_pct': m,
                                    'SEM_pct': s})
            if mech_id == 'LR' and 0 in xs:
                idx0 = int(np.flatnonzero(xs == 0)[0])
                ff_origin_by_occ[occ_key] = (means[idx0], sems[idx0])

        if human_df is not None:
            for group, color, marker, dashes, label in HUMAN_GROUPS:
                row = human_df.loc[(human_df['group'] == group)
                                   & (human_df['occlusion_pct'] == occ_pct)]
                if row.empty:
                    continue
                h_mean = float(row['accuracy_pct'].iloc[0])
                h_err = float(row['error_half_width'].iloc[0])
                human_by_occ[occ_key][group] = (h_mean, h_err)
                all_means.append(np.array([h_mean]))
                all_sems.append(np.array([h_err]))
                export_rows.append({'Mechanism': f'Human ({group.lower()})', 'Stage': np.nan,
                                    'Occlusion_pct': occ_pct, 'Mean_Accuracy_pct': h_mean,
                                    'SEM_pct': h_err})

    if not all_means:
        log.error('No data found')
        return

    max_stage = int(agg['stage_idx'].max())
    all_means_c = np.concatenate(all_means)
    all_sems_c = np.concatenate(all_sems)
    y_lo = max(0.0, np.floor((all_means_c - all_sems_c).min() / 5.0) * 5.0 - 2.0)
    y_hi = max(101.5, np.ceil((all_means_c + all_sems_c).max() / 5.0) * 5.0 + 1.0)

    fig = plt.figure(figsize=FIGSIZE, dpi=110, facecolor=GROUND)
    gs = fig.add_gridspec(1, len(OCCLUSION_LEVELS), **GRID)

    legend_items = [
        (disp_label, color, marker, fs.MS_LEGEND if marker != 'D' else fs.MS_LEGEND_SM,
         None, 0)
        for (_, disp_label, short, color, marker, lw) in MECHANISMS
    ] + [(FF_LABEL, FF_COLOR, FF_MARKER, fs.MS_LEGEND, None, 0)] + [
        (label, color, marker, fs.MS_LEGEND, (0, dashes), fs.pt(1.1))
        for (_, color, marker, dashes, label) in HUMAN_GROUPS
    ]
    draw_legend_row(fig, legend_items, LEGEND_Y, GRID['left'], x_max=GRID['right'],
                    dy=LEGEND_DY)

    col_title_x = []
    for col_idx, (occ_key, occ_label, occ_pct) in enumerate(OCCLUSION_LEVELS):
        ax = fig.add_subplot(gs[0, col_idx])
        col_title_x.append(ax.get_position().x0)

        ax.axvline(0, color=HAIRLINE, linewidth=fs.LW_GUIDE, linestyle=(0, (1, 2.4)), zorder=1)

        # Human accuracy has no stage dimension: flat reference bands across the stage axis
        xs_full = np.array([-0.4, max_stage + 0.4])
        for group, color, marker, dashes, label in HUMAN_GROUPS:
            hv = human_by_occ[occ_key].get(group)
            if hv is None:
                continue
            h_mean, h_err = hv
            draw_band(ax, xs_full, [h_mean - h_err] * 2, [h_mean + h_err] * 2, color,
                      fill_alpha=0.12, zorder=1.5)
            ax.plot(xs_full, [h_mean, h_mean], color=color, linewidth=fs.pt(1.1),
                    linestyle='--', dashes=dashes, zorder=3)

        for mech_id, disp_label, short, color, marker, lw in MECHANISMS:
            s = series_by_occ[occ_key].get(mech_id)
            if s is None:
                continue
            draw_band(ax, s['xs'], s['means'] - s['sems'], s['means'] + s['sems'], color)
            ax.plot(s['xs'], s['means'], color=color, linewidth=lw, solid_capstyle='round',
                    marker=marker, markersize=fs.MS_PEAK if marker != 'D' else fs.MS_PEAK_SM,
                    markeredgecolor=GROUND, markeredgewidth=fs.pt(0.65), zorder=6)

        # Shared feedforward origin (stage 0 is identical for LR and TD)
        if occ_key in ff_origin_by_occ:
            ff_mean, _ff_sem = ff_origin_by_occ[occ_key]
            ax.plot([0], [ff_mean], marker=FF_MARKER, markersize=fs.MS_PEAK * 1.75,
                    markerfacecolor='none', markeredgecolor=FF_COLOR,
                    markeredgewidth=fs.pt(1.10), linestyle='none', zorder=7)

        ax.set_xlim(-0.4, max_stage + 0.4)
        ax.set_xticks(range(max_stage + 1))
        ax.set_ylim(y_lo, y_hi)
        style_axes(ax)
        ax.set_xlabel('Readout stage', fontsize=fs.FS_AXLABEL, color=INK, labelpad=fs.pt(1.8))
        if col_idx == 0:
            ax.set_ylabel('Accuracy (%)', fontsize=fs.FS_AXLABEL, color=INK,
                          labelpad=fs.pt(1.8))
        else:
            ax.tick_params(labelleft=False)

    for x0, (occ_key, occ_label, occ_pct) in zip(col_title_x, OCCLUSION_LEVELS):
        fig.text(x0, COLTITLE_Y, occ_label, fontsize=fs.FS_COLTITLE, fontweight='bold',
                 color=INK, ha='left', va='bottom')

    paths = save_figure(fig, out_dir, 'Figure_HTRN_StageSweep')
    report_page_fit(FIGSIZE, 'Figure_HTRN_StageSweep')
    plt.close(fig)
    log.info('Saved %s', paths[0])

    pd.DataFrame(export_rows).to_csv(os.path.join(out_dir, 'StageSweep_Data.csv'), index=False)


if __name__ == '__main__':
    main()
