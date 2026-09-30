"""Schematic of HTRN readout stages (Fig. 4A-B).

Panel A (fold / unfold): a ResNet bottleneck block is one local recurrent iteration. Unfolding the
self-loop of a unit h gives the chain h0 -> h1 -> h2. Two block states are used in panel B:
    active         h1 = h0 + K1(h0)   full residual computation (local recurrence)
    shortcut only  h1 = ReLU(W h0)    projection shortcut plus ReLU (feedforward skeleton)
Every readout stage still traverses all 16 blocks, so stage 0 (shortcut only) is a feedforward
sweep of the same network, not a shallower one.

Panel B (the two sweeps): rows are the 16 bottleneck blocks of ResNet-50 ([3, 4, 6, 3], grouped by
layer and mapped to V1-3 / LOC / IT-PHC), columns are readout stages. Both sweeps use the same block
schedule; they differ only in routing:
    HTRN-LR  stages 0-6  every stage is an independent pass from the frontend output, no feedback
    HTRN-TD  stages 0-7  local depth and top-down feedback grow together: at every stage >= 1 the
                         previous stage's layer4 output updates the running feedback state
                         (exponential decay, T = stage), which is gated onto the frontend output
                         before that stage's blocks run
Stage 7 adds one more feedback iteration at full local depth. An open FF ring marks stage 0, which
is identical in both sweeps. The input chevron is repeated under every column because each stage
re-runs the whole cascade from the input.

The figure needs no input data. Text stays editable in SVG (no mathtext; subscripts use Unicode
characters or a second text artist).

Output: results_htrn_unrolling/Figure_HTRN_Unrolling (PNG/SVG/PDF)

Usage:
    python scripts/07_figures/plot_htrn_unrolling.py
"""

import logging
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

from htrn import figstyle as fs, setup_logging
from htrn.config import DATA_ROOT
from htrn.figstyle import (
    INK, MID, FAINT, HAIRLINE, TRACK, GROUND,
    setup_rcparams, save_figure, report_page_fit,
)
from htrn.models import RESNET50_BLOCKS, generate_stage_schedule

log = logging.getLogger(__name__)


# --- Configuration ---

RESULTS_DIR = str(DATA_ROOT / 'results_htrn_unrolling')
STEM = 'Figure_HTRN_Unrolling'

# Bottleneck blocks per ResNet-50 layer
CAPS = list(RESNET50_BLOCKS)

LAYER_LABELS = ['Conv 2', 'Conv 3', 'Conv 4', 'Conv 5']
# Conv 2 and Conv 3 share V1-3, so the ROI tier is an outer bracket
ROI_GROUPS = [([0, 1], 'V1–3'), ([2], 'LOC'), ([3], 'IT–PHC')]

LR_MAX_STAGE = 6        # last local-recurrence stage
TD_MAX_STAGE = 7        # last top-down stage

# Variant colours and markers, as in the RSA and stage-sweep figures
FF_COLOR, FF_MARKER = '#F4256D', 'o'
LR_COLOR, LR_MARKER = '#3A0CA3', 's'
TD_COLOR, TD_MARKER = '#00A5A8', 'D'

FIGSIZE = (fs.FIG_W, 9.0)
AX_A = [0.012, 0.040, 0.269, 0.796]
AX_B = [0.297, 0.040, 0.688, 0.796]
LEGEND_Y = 0.972
TITLE_Y = 0.926


# --- Drawing primitives ---

def rbox(ax, x, y, w, h, facecolor, edgecolor, lw, zorder=3, rounding=None, alpha=1.0):
    """Rounded rectangle centred on (x, y), with the corner radius given in x-data units."""
    if rounding is None:
        rounding = min(w, h) * 0.30
    patch = FancyBboxPatch(
        (x - w / 2 + rounding, y - h / 2 + rounding),
        w - 2 * rounding, h - 2 * rounding,
        boxstyle=f'round,pad={rounding:.5f},rounding_size={rounding:.5f}',
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=alpha,
        mutation_aspect=1.0, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, p0, p1, color, lw, zorder=4, head=6.0):
    """Arrow from p0 to p1 with a filled head."""
    a = FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=head, color=color,
                        linewidth=lw, zorder=zorder, shrinkA=0, shrinkB=0,
                        joinstyle='round', capstyle='round')
    ax.add_patch(a)
    return a


def line(ax, xs, ys, color, lw, zorder=4):
    """Rounded polyline."""
    ax.plot(xs, ys, color=color, linewidth=lw, zorder=zorder, solid_capstyle='round',
            solid_joinstyle='round')


def plus_node(ax, x, y, rx, ry, color, lw, zorder=6):
    """The summation node. rx/ry are separate so it stays round on non-square axes."""
    ax.add_patch(Ellipse((x, y), 2 * rx, 2 * ry, facecolor=GROUND, edgecolor=color,
                         linewidth=lw, zorder=zorder))
    line(ax, [x - rx * 0.50, x + rx * 0.50], [y, y], color, lw, zorder + 1)
    line(ax, [x, x], [y - ry * 0.50, y + ry * 0.50], color, lw, zorder + 1)


def state_node(ax, x, y, r, label, color, lw, fontsize, zorder=6):
    """Labelled circle for a unit state."""
    ax.add_patch(Circle((x, y), r, facecolor=GROUND, edgecolor=color, linewidth=lw,
                        zorder=zorder))
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, color=INK,
            fontweight='bold', zorder=zorder + 1)


def sub_label(ax, cx, cy, base, sub, color, fontsize, dx, dy, zorder=7):
    """A base letter plus a dropped subscript, as two text artists (not mathtext)."""
    ax.text(cx - dx, cy, base, ha='right', va='center', fontsize=fontsize, color=color,
            fontweight='bold', zorder=zorder)
    ax.text(cx - dx, cy - dy, sub, ha='left', va='center', fontsize=fontsize * 0.76,
            color=color, fontweight='bold', zorder=zorder)


def bracket(ax, x, y0, y1, tip, color, lw, zorder=3):
    """Square bracket spanning y0..y1 at x, with stubs reaching out to `tip`."""
    line(ax, [x, x], [y0, y1], color, lw, zorder)
    for y in (y0, y1):
        line(ax, [x, tip], [y, y], color, lw, zorder)


# --- Panel A: fold / unfold ---

def draw_panel_a(ax):
    """Equal-aspect schematic: x runs 0-100 and y runs 0-ytop (from the axes size), so circles are round."""
    w_in = AX_A[2] * FIGSIZE[0]
    h_in = AX_A[3] * FIGSIZE[1]
    ytop = 100.0 * h_in / w_in

    ax.set_xlim(0, 100)
    ax.set_ylim(0, ytop)
    ax.set_aspect('equal')
    ax.axis('off')

    lw_thin = fs.pt(0.85)
    lw_mid = fs.pt(1.20)
    f_sm = fs.pt(6.0)
    f_bd = fs.pt(6.9)
    f_hd = fs.pt(7.5)

    spine, branch = 34.0, 71.0
    gap = 1.1          # arrowheads stop short of a node
    kw, kh = 25.0, 10.5

    # Folded
    ax.text(1, ytop - 1.5, 'Folded', fontsize=f_hd, color=INK, fontweight='bold',
            ha='left', va='top')

    h_y, plus_y, k_y = 127.0, 148.0, 137.5
    r_h, r_p = 7.0, 4.5

    arrow(ax, (spine, 116.0), (spine, h_y - r_h - gap), MID, lw_thin, head=5.5)
    ax.text(spine - 3.0, 117.5, 'input', fontsize=f_sm, color=MID, ha='right', va='center')
    state_node(ax, spine, h_y, r_h, 'h', MID, lw_mid, f_bd)

    # Shortcut (feedforward limb)
    arrow(ax, (spine, h_y + r_h), (spine, plus_y - r_p - gap), FF_COLOR, lw_mid, head=5.5)
    ax.text(spine - r_h - 3.6, (h_y + r_h + plus_y - r_p) / 2.0, 'shortcut', fontsize=f_sm,
            color=FF_COLOR, fontweight='bold', ha='center', va='center', rotation=90)

    # Recurrent limb through the block kernel
    line(ax, [spine + r_h, branch], [h_y, h_y], LR_COLOR, lw_mid)
    arrow(ax, (branch, h_y), (branch, k_y - kh / 2), LR_COLOR, lw_mid, head=5.5)
    rbox(ax, branch, k_y, kw, kh, GROUND, LR_COLOR, lw_mid, rounding=2.6)
    sub_label(ax, branch, k_y, 'K', 't', LR_COLOR, f_bd, dx=-0.9, dy=1.6)
    line(ax, [branch, branch], [k_y + kh / 2, plus_y], LR_COLOR, lw_mid)
    arrow(ax, (branch, plus_y), (spine + r_p + gap, plus_y), LR_COLOR, lw_mid, head=5.5)

    plus_node(ax, spine, plus_y, r_p, r_p, MID, lw_mid)
    arrow(ax, (spine, plus_y + r_p), (spine, ytop - 5.0), MID, lw_thin, head=5.5)

    # Unfold cue
    arrow(ax, (spine, 112.0), (spine, 105.0), FAINT, fs.pt(1.5), head=9.0)
    ax.text(spine + 4.0, 108.5, 'Unfold', fontsize=f_sm, color=MID, fontweight='bold',
            ha='left', va='center', style='italic')

    # Unrolled
    ax.text(1, 100.0, 'Unrolled', fontsize=f_hd, color=INK, fontweight='bold',
            ha='left', va='top')

    chain = [36.0, 58.0, 80.0]
    r_s, r_ps = 6.0, 4.0

    arrow(ax, (spine, 25.0), (spine, chain[0] - r_s - gap), MID, lw_thin, head=5.5)
    ax.text(spine - 3.0, 26.5, 'input', fontsize=f_sm, color=MID, ha='right', va='center')

    for i, y in enumerate(chain):
        state_node(ax, spine, y, r_s, f'h{"₀₁₂"[i]}', MID, lw_mid, f_sm)

    for i in range(2):
        y0, y1 = chain[i], chain[i + 1]
        py = (y0 + y1) / 2.0
        arrow(ax, (spine, y0 + r_s), (spine, py - r_ps - gap), FF_COLOR, lw_mid, head=5.0)
        arrow(ax, (spine, py + r_ps), (spine, y1 - r_s - gap), MID, lw_thin, head=5.0)
        line(ax, [spine + r_s, branch], [y0, y0], LR_COLOR, lw_mid)
        arrow(ax, (branch, y0), (branch, py - kh * 0.42), LR_COLOR, lw_mid, head=5.0)
        rbox(ax, branch, py, kw * 0.94, kh * 0.84, GROUND, LR_COLOR, lw_mid, rounding=2.3)
        sub_label(ax, branch, py, 'K', f'{i + 1}', LR_COLOR, f_sm, dx=-0.8, dy=1.4)
        arrow(ax, (branch - kw * 0.47, py), (spine + r_ps + gap, py), LR_COLOR, lw_mid,
              head=5.0)
        plus_node(ax, spine, py, r_ps, r_ps, MID, lw_mid)

    arrow(ax, (spine, chain[-1] + r_s), (spine, chain[-1] + r_s + 6.0), MID, lw_thin, head=5.5)
    ax.text(spine + 3.0, chain[-1] + r_s + 4.5, 'readout', fontsize=f_sm, color=MID,
            ha='left', va='center')

    # Readout-stage axis, linking iterations here to the columns of panel B
    ax.annotate('', xy=(8.0, 88.0), xytext=(8.0, 30.0),
                arrowprops=dict(arrowstyle='-|>', color=FAINT, linewidth=fs.pt(1.0),
                                mutation_scale=8.0, shrinkA=0, shrinkB=0))
    ax.text(5.4, 59.0, 'Readout stage', fontsize=f_sm, color=MID, fontweight='bold',
            rotation=90, ha='center', va='center')

    # The two block states
    cell_w, cell_h = 6.4, 3.4
    for y_row, filled, eq, cap in (
        (17.0, True, 'h₁ = h₀ + K₁(h₀)', 'block active — one recurrent iteration'),
        (4.0, False, 'h₁ = ReLU(W h₀)', 'shortcut only — the feedforward skeleton'),
    ):
        if filled:
            rbox(ax, 6.0, y_row, cell_w, cell_h, LR_COLOR, 'none', 0, rounding=1.0)
        else:
            rbox(ax, 6.0, y_row, cell_w, cell_h, GROUND, FF_COLOR, fs.pt(0.80), rounding=1.0)
        ax.text(12.5, y_row, eq, fontsize=f_bd, color=INK, fontweight='bold',
                ha='left', va='center')
        ax.text(12.5, y_row - 5.4, cap, fontsize=f_sm, color=MID, ha='left', va='center')


# --- Panel B: the two sweeps ---

GROUP_GAP = 0.85            # blank rows between two layers' block groups
BETWEEN_GROUPS = 2.5        # blank columns between the LR and the TD sweep


def _row_geometry():
    """Row centres of the 16 blocks, grouped by layer (shared by both sweeps)."""
    row_y, span = [], []
    y = 0.0
    for cap in CAPS:
        ys = [y + i for i in range(cap)]
        row_y.append(ys)
        span.append((ys[0], ys[-1]))
        y = ys[-1] + 1.0 + GROUP_GAP
    return row_y, span, span[-1][1]


def _draw_sweep(ax, x0, n_cols, schedule, row_y, top, g, feedback):
    """Draw one sweep: column rails, the block-by-stage cell matrix, input and readout rails and,
    for the top-down sweep, the per-stage feedback path.
    """
    lw_thin, lw_mid, f_sm = g['lw_thin'], g['lw_mid'], g['f_sm']
    y_in, y_out, y_fb = g['y_in'], g['y_out'], g['y_fb']
    r_x, r_y = g['r_x'], g['r_y']
    last = n_cols - 1
    terminal = TD_COLOR if feedback else LR_COLOR

    # Column rails; stage 0 and the terminal stage carry a variant tint
    for c in range(n_cols):
        col = FF_COLOR if c == 0 else (terminal if c == last else None)
        rbox(ax, x0 + c, top / 2.0, g['rail_w'], top + 1.80,
             facecolor=col if col else TRACK, edgecolor='none', lw=0,
             zorder=1, rounding=0.16, alpha=0.085 if col else 1.0)

    # Cell matrix from the block schedule (identical for both sweeps)
    for c in range(n_cols):
        counts = schedule[c]
        for li, cap in enumerate(CAPS):
            for bi in range(cap):
                if bi < counts[li]:
                    rbox(ax, x0 + c, row_y[li][bi], g['cell_w'], g['cell_h'], LR_COLOR,
                         'none', 0, zorder=3, rounding=0.11)
                else:
                    rbox(ax, x0 + c, row_y[li][bi], g['cell_w'], g['cell_h'], GROUND,
                         FF_COLOR, fs.pt(0.70), zorder=3, rounding=0.11)

    # Feedforward drive: the input is repeated under every column (each stage restarts from it)
    line(ax, [x0 - 0.62, x0 + last + 0.62], [y_in, y_in], HAIRLINE, lw_thin, zorder=2)
    line(ax, [x0 - 0.62, x0 + last + 0.62], [y_out, y_out], HAIRLINE, lw_thin, zorder=2)
    for c in range(n_cols):
        x = x0 + c
        if feedback and c >= 1:
            arrow(ax, (x, y_in), (x, y_fb - r_y), MID, lw_thin, head=5.0)
            plus_node(ax, x, y_fb, r_x, r_y, TD_COLOR, lw_mid)
            arrow(ax, (x, y_fb + r_y), (x, -0.92), TD_COLOR, lw_mid, head=5.5)
        else:
            arrow(ax, (x, y_in), (x, -0.92), MID, lw_thin, head=5.0)
        arrow(ax, (x, top + 0.92), (x, y_out), MID, lw_thin, head=5.0)

    # Top-down path: stage k-1's layer4 readout -> accumulated state -> gate -> summed with the
    # frontend output at stage k's input, before that stage's blocks run
    if feedback:
        hop = y_out + 0.62
        for c in range(1, n_cols):
            x_src, x_dst = x0 + c - 1, x0 + c
            mid = (x_src + x_dst) / 2.0
            line(ax, [x_src, x_src, mid, mid, x_dst - r_x - 0.12],
                 [y_out, hop, hop, y_fb, y_fb], TD_COLOR, lw_mid, zorder=5)
            arrow(ax, (x_dst - r_x - 0.12, y_fb), (x_dst - r_x, y_fb), TD_COLOR, lw_mid,
                  head=6.0, zorder=5)

    for c in range(n_cols):
        ax.text(x0 + c, g['y_num'], str(c), fontsize=f_sm, color=MID, ha='center', va='center')


def draw_panel_b(ax, schedule):
    """Both sweeps side by side with shared block labels and ROI brackets."""
    lw_thin = fs.pt(0.75)
    lw_mid = fs.pt(1.15)
    f_sm = fs.pt(6.0)
    f_md = fs.pt(6.8)

    row_y, span, top = _row_geometry()

    n_lr, n_td = LR_MAX_STAGE + 1, TD_MAX_STAGE + 1
    x_lr = 0.0
    x_td = x_lr + (n_lr - 1) + BETWEEN_GROUPS

    x_lab = -3.30
    x_lo, x_hi = x_lab - 0.15, x_td + (n_td - 1) + 0.70
    y_lo, y_hi = -3.70, top + 6.10
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.axis('off')

    # Data units per inch, to keep the summation nodes round
    upi_x = (x_hi - x_lo) / (AX_B[2] * FIGSIZE[0])
    upi_y = (y_hi - y_lo) / (AX_B[3] * FIGSIZE[1])
    r_node = 0.10   # inches

    g = dict(
        lw_thin=lw_thin, lw_mid=lw_mid, f_sm=f_sm,
        y_in=-2.15, y_out=top + 1.55, y_fb=-1.32, y_num=-2.95,
        cell_w=0.40, cell_h=0.56, rail_w=0.78,
        r_x=r_node * upi_x, r_y=r_node * upi_y,
    )

    _draw_sweep(ax, x_lr, n_lr, schedule, row_y, top, g, feedback=False)
    _draw_sweep(ax, x_td, n_td, schedule, row_y, top, g, feedback=True)

    # Left labels: layer name and block count, then the ROI bracket
    for li, cap in enumerate(CAPS):
        y0, y1 = span[li]
        ax.text(-0.80, (y0 + y1) / 2.0, f'{LAYER_LABELS[li]}  ×{cap}', fontsize=f_sm,
                color=INK, fontweight='bold', ha='right', va='center')
    for rows, roi in ROI_GROUPS:
        y0 = span[rows[0]][0] - 0.45
        y1 = span[rows[-1]][1] + 0.45
        bracket(ax, x_lab + 0.34, y0, y1, x_lab + 0.56, FAINT, lw_thin)
        ax.text(x_lab + 0.16, (y0 + y1) / 2.0, roi, fontsize=f_sm, color=MID,
                fontweight='bold', ha='center', va='center', rotation=90)

    ax.text(-0.80, g['y_out'], 'readout', fontsize=f_sm, color=MID, ha='right', va='center')
    ax.text(-0.80, g['y_in'], 'input', fontsize=f_sm, color=MID, ha='right', va='center')
    ax.text(-0.80, g['y_num'], 'Readout stage', fontsize=fs.FS_AXLABEL, color=INK,
            fontweight='bold', ha='right', va='center')

    # Sweep titles
    for x0, n_cols, color, title, sub in (
        (x_lr, n_lr, LR_COLOR, 'HTRN-LR', 'local recurrence only'),
        (x_td, n_td, TD_COLOR, 'HTRN-TD', 'local recurrence + top-down feedback'),
    ):
        cx = x0 + (n_cols - 1) / 2.0
        ax.text(cx, top + 5.35, title, fontsize=fs.FS_COLTITLE, color=color,
                fontweight='bold', ha='center', va='center')
        ax.text(cx, top + 4.55, sub, fontsize=f_md, color=MID, ha='center', va='center')

    # Markers for the three variants; stage 0 is shared, so the FF ring sits on column 0 of both
    for x, label, color, marker, hollow in (
        (x_lr, 'HTRN-FF', FF_COLOR, FF_MARKER, True),
        (x_lr + LR_MAX_STAGE, 'HTRN-LR', LR_COLOR, LR_MARKER, False),
        (x_td, 'HTRN-FF', FF_COLOR, FF_MARKER, True),
        (x_td + TD_MAX_STAGE, 'HTRN-TD', TD_COLOR, TD_MARKER, False),
    ):
        ms = fs.MS_PEAK_SM if marker == 'D' else fs.MS_PEAK
        ax.plot([x], [top + 3.05], marker=marker, linestyle='none',
                markersize=ms * (1.55 if hollow else 1.0),
                markerfacecolor='none' if hollow else color,
                markeredgecolor=color, markeredgewidth=fs.pt(1.10 if hollow else 0.65),
                zorder=8)
        ax.text(x, top + 3.72, label, fontsize=f_sm, color=color, fontweight='bold',
                ha='center', va='center')


# --- Legend ---

def draw_legend(fig, y, x0):
    """Legend row with hollow / filled block swatches (a marker cannot be hollow in draw_legend_row)."""
    ax = fig.add_axes([0, 0, 1, 1], facecolor='none', zorder=20)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_navigate(False)

    sw_w, sw_h = 0.0115, 0.0125
    x = x0
    for label, color, filled in (
        ('Recurrent iteration (block active)', LR_COLOR, True),
        ('Shortcut only (feedforward)', FF_COLOR, False),
        ('Top-down feedback', TD_COLOR, None),
    ):
        if filled is None:
            line(ax, [x, x + sw_w * 1.7], [y, y], color, fs.pt(1.5), zorder=21)
            ax.plot([x + sw_w * 0.85], [y], marker=TD_MARKER, color=color, linestyle='none',
                    markersize=fs.MS_LEGEND_SM, markeredgecolor=GROUND,
                    markeredgewidth=fs.pt(0.65), zorder=22)
            text_x = x + sw_w * 1.7 + 0.008
        else:
            rbox(ax, x + sw_w / 2, y, sw_w, sw_h,
                 color if filled else GROUND, 'none' if filled else color,
                 0 if filled else fs.pt(0.80), zorder=21, rounding=0.0035)
            text_x = x + sw_w + 0.008
        t = ax.text(text_x, y, label, fontsize=fs.FS_LEGEND, color=INK, fontweight='bold',
                    ha='left', va='center')
        fig.canvas.draw()
        w = t.get_window_extent(fig.canvas.get_renderer()) \
             .transformed(fig.transFigure.inverted()).width
        x = text_x + w + 0.026


def main():
    """Draw the schematic."""
    setup_logging()
    setup_rcparams()

    schedule = generate_stage_schedule(CAPS, n_tail_stages=5)
    assert schedule[0] == [0, 0, 0, 0], 'stage 0 must activate no residual block'
    assert schedule[LR_MAX_STAGE] == CAPS, 'LR_MAX_STAGE must be full local-recurrent depth'
    assert schedule[TD_MAX_STAGE] == schedule[LR_MAX_STAGE], \
        'TD stage 7 must add feedback, not depth - its block counts must equal stage 6'

    fig = plt.figure(figsize=FIGSIZE, dpi=110, facecolor=GROUND)
    draw_panel_a(fig.add_axes(AX_A))
    draw_panel_b(fig.add_axes(AX_B), schedule)

    for x_letter, x_title, letter, title in (
        (0.006, 0.030, 'A', 'A block is one recurrent iteration'),
        (0.281, 0.305, 'B', 'How the readout stages are built'),
    ):
        fig.text(x_letter, TITLE_Y, letter, fontsize=fs.FS_COLTITLE, fontweight='bold',
                 color=INK, ha='left', va='center')
        fig.text(x_title, TITLE_Y, title, fontsize=fs.FS_COLTITLE_SM, fontweight='bold',
                 color=INK, ha='left', va='center')

    draw_legend(fig, LEGEND_Y, 0.019)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    paths = save_figure(fig, RESULTS_DIR, STEM)
    report_page_fit(FIGSIZE, STEM)
    plt.close(fig)
    log.info('Saved %s', paths[0])


if __name__ == '__main__':
    main()
