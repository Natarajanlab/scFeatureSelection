#!/usr/bin/env python3
"""
make_geneset_figures_final.py
=============================
Reproduce all gene-set classifier performance figures locally.

FIGURES PRODUCED
----------------
For each dataset (3k, 10k):

  1. *_PRAUC_curves.pdf      — PR-AUC vs number of features (log x-axis)
                               Gaussian PDF shading from SD at each x position
  2. *_ROCAUC_curves.pdf     — Same design for ROC-AUC
  3. *_cleveland_ct.pdf      — Cleveland dot plot at fixed N (5, 20, 50 genes)
                               Plain circles sized by N landmark (small/med/large)
                               Error bars = SD from CSV; ±value printed at cap
                               n= label above each dot (deduplicated)
                               Two blocks: Leiden (bottom) | CellTypist (top)
  4. *_dotstrip_ct_gap.pdf   — Lollipop / gap chart at 5% & 10% vs 100% ref
                               Two blocks per panel with dashed gap separator
  5. *_dotstrip_ct_separate.pdf — Same data, Leiden and CellTypist as separate
                               row groups across 4 sub-panels

INPUT CSV FORMAT
----------------
One CSV per dataset × classification target. All files share the same schema:

  Columns (one block per gene set, four gene sets total):
    hk          — number of Housekeeping features used
    hk_pr       — PR-AUC mean
    hk_pr_std   — PR-AUC standard deviation (from cross-validation)
    hk_auc      — ROC-AUC mean
    hk_auc_std  — ROC-AUC standard deviation
    hvg, hvg_pr, hvg_pr_std, hvg_auc, hvg_auc_std
    nhvg, nhvg_pr, nhvg_pr_std, nhvg_auc, nhvg_auc_std
    uc, uc_pr, uc_pr_std, uc_auc, uc_auc_std

  Each row = one feature-count step. Rows where a gene set column is empty
  or zero are skipped (e.g. early rows before Union-Cluster features appear).

EDIT THE PATHS BELOW
--------------------
Set LEIDEN_FILES and CT_FILES to your local file paths, then run:

    python make_geneset_figures_final.py
    python make_geneset_figures_final.py --out ./my_figures

DEPENDENCIES
------------
    pip install matplotlib scipy numpy

Tested with:
    Python    >= 3.9
    matplotlib >= 3.7
    scipy      >= 1.10
    numpy      >= 1.24
"""

import argparse
import csv
import os
import numpy as np

import matplotlib
matplotlib.use('Agg')          # change to 'TkAgg' or remove for interactive use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import norm as scipy_norm


# ══════════════════════════════════════════════════════════════════════════════
# INPUT FILE PATHS  — edit these to point to your local CSV/TSV files
# ══════════════════════════════════════════════════════════════════════════════

LEIDEN_FILES = {
    # '3k': {
    #     'Leiden 0.02': 'pbmc3k_1000hvgs_leiden0_02.csv',
    #     'CellTypist':  'pbmc3k_celltypist_labels.csv',
    # },
    # '10k': {
    #     'Leiden 0.02': 'pbmc10k_1000hvgs_leiden0_02.csv',
    #     'Leiden 0.5':  'pbmc10k_1000hvgs_leiden0_5.csv',
    #     'Leiden 2.0':  'pbmc10k_1000hvgs_leiden2_0.csv',
    #     'CellTypist':  'pbmc10k_celltypist_labels.csv',
    # },
    # 'Zheng': {
    #     'distinct': 'zheng_pbmc_selected_celltypes.csv',
    #     'same':  'zheng_pbmc_cd4_only.csv',
    # },
    # 'Wu_Qian': {
    #     'train-Wu_test-Qian': 'train-Wu_2021_test-Qian_2020.csv',
    #     'train-Qian_test-Wu.csv':  'train-Qian_2020_test-Wu_2021.csv',
    # },
    'Sergio': {
        'Low Interaction': 'Sergio_5_10.csv',
        'Medium Interaction': 'Sergio_20_30.csv',
        'High Interaction' : 'Sergio_50_60.csv',
    },
}


# TSV support: set DELIMITER = '\t' if your files are tab-separated
DELIMITER = ','


# ══════════════════════════════════════════════════════════════════════════════
# APPEARANCE  — colours, markers, line styles, sizes
# ══════════════════════════════════════════════════════════════════════════════

# COLORS = {
#     'nhvg': '#1D9E75',   # teal   — Non-HVGs
#     'hvg':  '#378ADD',   # blue   — HVGs
#     'hk':   '#888780',   # grey   — Housekeeping
#     'uc':   '#D85A30',   # coral  — Union-Cluster
#     # 'fs': 'purple'
# }
COLORS = {
    'high': '#1D9E75',   # teal   — Non-HVGs
    'low':  '#378ADD',   # blue   — HVGs
}
# LABELS = {
#     'nhvg': 'Non-HVGs',
#     'hvg':  'HVGs',
#     'hk':   'Housekeeping',
#     'uc':   'Union-Cluster',
#     # 'fs': 'dubSTEPR'
# }
LABELS = {
    'high': 'High Separation',
    'low':  'Low Separation',
}
# Line dash patterns for curves (None = solid)
# DASH = {
#     'nhvg': None,
#     'hvg':  (5, 2),
#     'hk':   (2, 2),
#     'uc':   (8, 2, 2, 2),
#     # 'fs': (1, 2)
# }
DASH = {
    'high': None,
    'low':  (5, 2),
}
# Marker shapes for curves (each gene set gets a distinct shape)
# CURVE_MARKER = {'nhvg': 'o', 'hvg': 's', 'hk': '^', 'uc': 'D', 'fs': 'P'}
CURVE_MARKER = {'high': 'o', 'low': 's'}

# Cleveland + dotstrip always use plain circles; size encodes N landmark
DOT_MARKER = 'o'
DOT_SIZES  = [55, 120, 300]   # small (N≈5) → medium (N≈20) → large (N≈50)

# SETS    = ['nhvg', 'hvg', 'hk', 'uc']
SETS    = ['high', 'low']
FIXED_N = [5, 20, 50]         # fixed absolute feature counts for Cleveland

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        8,
    'axes.linewidth':   0.6,
    'axes.labelsize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size':  3,
    'ytick.major.size':  3,
    'legend.fontsize':  7.5,
    'legend.frameon':   False,
    'pdf.fonttype':     42,    # embed fonts (required for journal submission)
    'ps.fonttype':      42,
})


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_series(path, delimiter=','):
    """
    Load a CSV/TSV file and return per-gene-set arrays sorted by feature count.

    Returns
    -------
    dict : { gene_set_key -> {
                'n':      np.ndarray,   feature counts (sorted ascending)
                'pr':     np.ndarray,   PR-AUC mean
                'pr_sd':  np.ndarray,   PR-AUC standard deviation
                'auc':    np.ndarray,   ROC-AUC mean
                'auc_sd': np.ndarray,   ROC-AUC standard deviation
             } }
    """
    with open(path, newline='') as fh:
        rows = list(csv.DictReader(fh, delimiter=delimiter))

    out = {}
    for s in SETS:
        ns, prs, aucs, pr_sds, auc_sds = [], [], [], [], []
        for r in rows:
            try:
                n = float(r[s])
                if n <= 0:
                    continue
                ns.append(n)
                prs.append(float(r[f'{s}_pr']))
                aucs.append(float(r[f'{s}_auc']))
                pr_sds.append(float(r[f'{s}_pr_std']))
                auc_sds.append(float(r[f'{s}_auc_std']))
            except (ValueError, KeyError):
                pass   # skip rows where this gene set has no data yet

        if ns:
            arr = np.array([ns, prs, aucs, pr_sds, auc_sds])
            idx = np.argsort(arr[0])
            out[s] = {
                'n':      arr[0, idx],
                'pr':     arr[1, idx],
                'auc':    arr[2, idx],
                'pr_sd':  arr[3, idx],
                'auc_sd': arr[4, idx],
            }
    return out


def snap_fixed_n(series, fixed_ns=FIXED_N):
    """
    For each gene set, find the row whose feature count is closest to each
    value in fixed_ns.  If a target exceeds the gene set's maximum N, that
    entry is set to None.

    Returns
    -------
    dict : { gene_set_key -> { target_n -> {
                'n_actual': int,
                'pr':       float,   'pr_sd':  float,
                'auc':      float,   'auc_sd': float,
             } | None } }
    """
    out = {}
    for s, d in series.items():
        out[s] = {}
        for fn in fixed_ns:
            if fn > d['n'].max():
                out[s][fn] = None
                continue
            i = int(np.argmin(np.abs(d['n'] - fn)))
            out[s][fn] = {k: float(d[k][i])
                          for k in ['n', 'pr', 'auc', 'pr_sd', 'auc_sd']}
            out[s][fn]['n_actual'] = int(d['n'][i])
    return out


def get_pct_snaps(series):
    """
    For each gene set, find rows at 5%, 10%, and 100% of its OWN total N.

    Note: because each gene set has a different total, '5%' means different
    absolute numbers of features for each gene set.  The actual N used is
    stored in 'n_actual' and labelled explicitly in the figures.

    Returns
    -------
    dict : { gene_set_key -> { '5%' | '10%' | '100%' -> {
                'pr':      float,   'pr_sd':  float,
                'auc':     float,   'auc_sd': float,
                'n_actual': int,    'total':  int,
             } } }
    """
    out = {}
    for s, d in series.items():
        tot = d['n'].max()
        out[s] = {}
        for label, target in [('5%', tot * 0.05),
                               ('10%', tot * 0.10),
                               ('100%', tot)]:
            i = int(np.argmin(np.abs(d['n'] - target)))
            out[s][label] = {
                'pr':      d['pr'][i],
                'auc':     d['auc'][i],
                'pr_sd':   d['pr_sd'][i],
                'auc_sd':  d['auc_sd'][i],
                'n_actual': int(d['n'][i]),
                'total':    int(tot),
            }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 & 2 — CURVE PLOTS  (PR-AUC or ROC-AUC vs feature count)
# ══════════════════════════════════════════════════════════════════════════════

def _pdf_shading(ax, n_arr, mu_arr, sd_arr, color, n_xpts=16, n_ypts=120):
    """
    At each of n_xpts x-positions, draw a vertical Gaussian PDF using the
    mean (mu) and SD from the data.  Each PDF is rendered as stacked thin
    coloured line segments whose alpha scales with the PDF height, producing
    a smooth gradient that is widest where uncertainty is largest.
    """
    xi_sub = np.unique(
        np.round(np.linspace(0, len(n_arr) - 1, n_xpts)).astype(int)
    )
    for xi in xi_sub:
        mu, sd = mu_arr[xi], max(float(sd_arr[xi]), 1e-4)
        y_lo   = max(0.0, mu - 3 * sd)
        y_hi   = min(1.0, mu + 3 * sd)
        ys     = np.linspace(y_lo, y_hi, n_ypts)
        pdf    = scipy_norm.pdf(ys, mu, sd)
        pdf   /= pdf.max()
        for j in range(n_ypts - 1):
            alpha = float((pdf[j] + pdf[j + 1]) / 2) * 0.28
            if alpha > 0.005:
                ax.plot(
                    [n_arr[xi], n_arr[xi]], [ys[j], ys[j + 1]],
                    color=color, alpha=alpha, linewidth=2.2,
                    solid_capstyle='butt', zorder=1,
                )


def make_curve_single_fig(ser, dataset_label, file_label, out_path):
    """
    Save a curve plot for a single file/resolution.
    It contains 2 panels side-by-side:
      Left panel: ROC-AUC vs number of features (log scale)
      Right panel: PR-AUC vs number of features (log scale)
    Both panels show curves for all gene sets with Gaussian PDF shading from SD.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.4, 3.8),
        constrained_layout=True,
    )
    
    metrics = [('auc', 'ROC-AUC'), ('pr', 'PR-AUC')]
    
    for ax, (metric, mlabel) in zip(axes, metrics):
        n_max = max(ser[s]['n'].max() for s in SETS if s in ser)
        for s in SETS:
            if s not in ser:
                continue
            d = ser[s]
            c = COLORS[s]
            
            # Gaussian PDF shading
            _pdf_shading(ax, d['n'], d[metric], d[f'{metric}_sd'], c)
            
            # main curve
            lkw = dict(color=c, linewidth=1.8, zorder=3)
            if DASH[s]:
                lkw['dashes'] = DASH[s]
            ax.semilogx(
                d['n'], d[metric], **lkw,
                marker=CURVE_MARKER[s], markersize=3.5, markevery=3,
                markerfacecolor='white', markeredgewidth=1.0,
            )
            
        ax.axhline(0.9, color='#cccccc', linewidth=0.7, linestyle=':', zorder=0)
        ax.set_xlim(0.8, n_max * 1.4)
        ax.set_ylim(0, 1.06)
        ax.set_xlabel('Number of features (log scale)', fontsize=8)
        ax.set_ylabel(mlabel, fontsize=9)
        ax.set_title(mlabel, fontsize=9, fontweight='bold', pad=4)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True, which='both', linestyle='--', linewidth=0.3, color='#ccc', alpha=0.5)
        ax.spines[['top', 'right']].set_visible(False)
        
    leg = [
        Line2D([0], [0],
               color=COLORS[s], linewidth=1.6,
               dashes=DASH[s] if DASH[s] else (None, None),
               marker=CURVE_MARKER[s], markersize=4,
               markerfacecolor='white', markeredgewidth=0.9,
               label=LABELS[s])
        for s in SETS
    ]
    fig.legend(handles=leg, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.09), frameon=False,
               fontsize=7.5, handlelength=2.5, columnspacing=1.2)
    fig.suptitle(
        f'{dataset_label} ({file_label})  ·  Performance vs. number of features\n'
        f'(shading = Gaussian PDF from cross-validation SD)',
        fontsize=9.5, fontweight='bold', y=1.02,
    )
    fig.savefig(out_path, format='pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — CLEVELAND DOT PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _draw_cleveland_rows(ax, snap, y_fn, mkey):
    """
    Draw one block of gene-set rows on a Cleveland dot plot.
    Includes vertical staggering to prevent text/dot overlap.
    """
    for si, s in enumerate(SETS):
        if s not in snap:
            continue
        c  = COLORS[s]
        yp = y_fn(si)

        valid = [
            (fn, snap[s][fn])
            for fn in FIXED_N
            if snap[s].get(fn) is not None
        ]
        if not valid:
            continue

        # ── deduplicate landmarks that map to the same n_actual ───────────
        seen_n   = {}
        deduped  = []
        for fi, (fn, d) in enumerate(valid):
            na = d['n_actual']
            if na not in seen_n:
                seen_n[na] = fi
                deduped.append((fi, fn, d))
            else:
                # replace earlier entry with this (larger) dot
                deduped = [
                    (fi, fn, d) if entry[2]['n_actual'] == na else entry
                    for entry in deduped
                ]
                seen_n[na] = fi

        # Sort deduped landmarks by performance value (X-axis)
        deduped_sorted = sorted(deduped, key=lambda x: x[2][mkey])
        n_points = len(deduped_sorted)

        # Calculate vertical offsets to prevent overlap if points are close in X
        offsets = [0.0] * n_points
        if n_points == 2:
            val0 = deduped_sorted[0][2][mkey]
            val1 = deduped_sorted[1][2][mkey]
            if abs(val1 - val0) < 0.05:
                offsets = [-0.08, 0.08]
        elif n_points == 3:
            val0 = deduped_sorted[0][2][mkey]
            val1 = deduped_sorted[1][2][mkey]
            val2 = deduped_sorted[2][2][mkey]
            diff01 = abs(val1 - val0)
            diff12 = abs(val2 - val1)
            if diff01 < 0.05 and diff12 < 0.05:
                offsets = [-0.12, 0.0, 0.12]
            elif diff01 < 0.05:
                offsets = [-0.08, 0.08, 0.0]
            elif diff12 < 0.05:
                offsets = [0.0, -0.08, 0.08]

        # ── connecting line ───────────────────────────────────────────────
        x_vals = [d[mkey] for _, _, d in deduped_sorted]
        y_vals = [yp + off for off in offsets]
        if len(x_vals) > 1:
            ax.plot(x_vals, y_vals,
                    color=c, linewidth=0.9, alpha=0.28, zorder=1)

        # ── dots, error bars, labels ──────────────────────────────────────
        for idx, (fi, fn, d) in enumerate(deduped_sorted):
            val   = d[mkey]
            sd    = d[f'{mkey}_sd']   # exact SD from CSV
            n_act = d['n_actual']
            sz    = DOT_SIZES[fi]
            y_dot = yp + offsets[idx]

            # error bar (width = SD from CSV)
            ax.errorbar(
                val, y_dot, xerr=sd,
                fmt='none', color=c,
                linewidth=0.9, capsize=3, capthick=0.8,
                alpha=0.7, zorder=2,
            )
            # SD numeric value at right cap
            ax.text(
                val, y_dot - 0.28, f'±{sd:.2f}',
                ha='center', va='center',
                fontsize=5.4, color=c, alpha=0.75, 
                style='italic', fontweight='500', zorder=5,
            )
            # ax.text(
            #     val + sd + 0.005, y_dot,
            #     f'±{sd:.2f}',
            #     ha='left', va='center',
            #     fontsize=4.4, color=c, alpha=0.75,
            #     style='italic', zorder=5,
            # )


            # dot (plain circle)
            ax.scatter(
                val, y_dot, s=sz, c=c, zorder=4,
                marker=DOT_MARKER,
                edgecolors='white', linewidths=0.7, alpha=0.92,
            )

            # AUC value inside dot
            fc = '#333333'
            ax.text(
                val, y_dot, f'{val:.2f}',
                ha='center', va='center',
                fontsize=5.6, color=fc, fontweight='600', zorder=5,
            )

            # n= label above dot (fixed offset, staggered with dot)
            ax.text(
                val, y_dot + 0.28, f'n={n_act}',
                ha='center', va='center',
                fontsize=6.0, color=c, fontweight='500', zorder=5,
            )


def make_cleveland_plot(leiden_snaps, dataset_label, metric, out_path):
    """
    Cleveland dot plot with resolutions stacked vertically (1 column, N rows).
    All panels share the same X-axis.
    """
    res_labels = list(leiden_snaps.keys())
    n_res      = len(res_labels)
    metric_label = 'PR-AUC' if metric == 'pr' else 'ROC-AUC'

    # Compute shared x-limits
    all_vals, all_upper = [], []
    for rl in res_labels:
        snap = leiden_snaps[rl]
        for s in SETS:
            if s not in snap:
                continue
            for fn in FIXED_N:
                d = snap[s].get(fn)
                if d:
                    v, sd = d[metric], d[f'{metric}_sd']
                    all_vals.append(v)
                    all_upper.append(v + sd)
                    
    if all_vals:
        x_min = max(0.0,  min(all_vals)  - 0.14)
        x_max = min(1.00, max(all_upper)) + 0.12
    else:
        x_min = 0.0
        x_max = 1.0

    # Stack panels vertically: n_res rows, 1 column
    fig, axes = plt.subplots(
        n_res, 1,
        figsize=(4.8, 2.5 * n_res + 0.8),
        constrained_layout=True,
        sharex=True,
    )
    if n_res == 1:
        axes = [axes]

    def y_fn(si):
        return float(len(SETS) - 1 - si)

    y_min = -0.65
    y_max = len(SETS) - 0.35

    for ci, rl in enumerate(res_labels):
        ax = axes[ci]
        l_snap = leiden_snaps[rl]

        # reference grid
        for xr in [0.5, 0.75, 1.0]:
            if x_min <= xr <= x_max:
                ax.axvline(xr, color='#ebebeb', linewidth=0.7,
                           linestyle='--', zorder=0)
        ax.axvline(0.9, color='#cccccc', linewidth=0.8,
                   linestyle=':', zorder=0)

        # draw rows
        _draw_cleveland_rows(ax, l_snap, y_fn, metric)

        # within-block row separators
        for si in range(len(SETS) - 1):
            ax.axhline(y_fn(si) - 0.5,
                       color='#f0f0f0', linewidth=0.5, zorder=0)

        # axes limits and formatting
        ax.set_xlim(x_min, x_max + 0.06)
        ax.set_ylim(y_min, y_max)
        tick_ys = [y_fn(i) for i in range(len(SETS))]
        tick_labels = [LABELS[s] for s in SETS]
        ax.set_yticks(tick_ys)
        ax.set_yticklabels(tick_labels, fontsize=7.8)
        
        # Label the panel on the right side
        ax.text(
            x_max + 0.08,
            np.mean(tick_ys),
            rl, ha='left', va='center',
            fontsize=8, fontweight='bold', color='#444',
            rotation=270,
        )
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.tick_params(left=True, labelsize=7.5)

    # Set x-label on the bottom panel
    axes[-1].set_xlabel(metric_label, fontsize=8.5, labelpad=3)

    leg_dots = [
        plt.scatter([], [], s=DOT_SIZES[i], c='#888780', alpha=0.90,
                    edgecolors='white', linewidths=0.7, marker=DOT_MARKER,
                    label=f'N ≈ {FIXED_N[i]}  (exact n printed above dot)')
        for i in range(len(FIXED_N))
    ]
    fig.legend(
        handles=leg_dots,
        title=(
            'Fixed N landmark  ·  AUC value inside dot  ·  '
            'error bar = SD from CSV  (±value at cap)'
        ),
        title_fontsize=6.5,
        loc='lower center', ncol=3,
        bbox_to_anchor=(0.5, -0.07),
        frameon=True, framealpha=0.95, edgecolor='#ddd',
        fontsize=7, handletextpad=0.6, columnspacing=1.2,
    )
    fig.suptitle(
        f'{dataset_label}  ·  {metric_label} at fixed feature counts'
        f'  (N = {", ".join(str(x) for x in FIXED_N)} genes)\n'
        f'error bar = SD from cross-validation',
        fontsize=9.5, fontweight='bold', y=1.02,
    )
    fig.savefig(out_path, format='pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 & 5 — DOTSTRIP LOLLIPOP CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_dotstrip_rows(ax, psnap, y_fn, mkey):
    """
    Draw lollipop rows for one classification target and one metric.
    Includes overlap prevention for dot labels with the 100% reference bar.
    """
    for s in SETS:
        if s not in psnap:
            continue
        c   = COLORS[s]
        yp  = y_fn(s)
        d   = psnap[s]
        ref = d['100%'][mkey]
        v5  = d['5%'][mkey];  n5  = d['5%']['n_actual']
        v10 = d['10%'][mkey]; n10 = d['10%']['n_actual']
        tot = d['100%']['total']

        # 100% reference bar (faint fill + edge line)
        ax.barh(yp, ref, height=0.36, color=c, alpha=0.10,
                left=0, linewidth=0, zorder=1)
        ax.plot([ref, ref], [yp - 0.18, yp + 0.18],
                color=c, linewidth=1.1, alpha=0.35, zorder=2)

        # lollipop stems
        ax.plot([0, v5],  [yp - 0.09, yp - 0.09],
                color=c, linewidth=0.6, alpha=0.38, zorder=2)
        ax.plot([0, v10], [yp + 0.09, yp + 0.09],
                color=c, linewidth=0.6, alpha=0.38, zorder=2)

        # dots
        ax.scatter(v5,  yp - 0.09, s=55,  c=c, zorder=4,
                   marker='o', edgecolors='white',
                   linewidths=0.7, alpha=0.95)
        ax.scatter(v10, yp + 0.09, s=120, c=c, zorder=4,
                   marker='o', edgecolors='white',
                   linewidths=0.7, alpha=0.95)

        # gap arrows (dot → 100% line)
        for v, yoff in [(v5, yp - 0.09), (v10, yp + 0.09)]:
            if ref - v > 0.02:
                ax.annotate(
                    '', xy=(ref - 0.005, yoff), xytext=(v + 0.005, yoff),
                    arrowprops=dict(arrowstyle='-|>', color=c,
                                   alpha=0.22, lw=0.7, mutation_scale=5),
                )

        # Move n to the right of the y-axis, adding brackets
        ax.text(+0.02, yp - 0.09 - 0.16, f'n={n5} (5%)',
                ha='left', va='center', fontsize=7.0, color=c,
                fontweight='500', clip_on=False, zorder=5)
        ax.text(+0.02, yp + 0.09 + 0.16, f'n={n10} (10%)',
                ha='left', va='center', fontsize=7.0, color=c,
                fontweight='500', clip_on=False, zorder=5)

        # value labels (prevent overlap with reference label at right edge)
        # 5% value is placed at the second line (original n position)
        if ref - v5 < 0.045:
            ha_5 = 'right'
            x_5 = v5 - 0.015
        else:
            ha_5 = 'center'
            x_5 = v5

        # 10% value is placed at the bottom of the text block (original n position)
        if ref - v10 < 0.045:
            ha_10 = 'right'
            x_10 = v10 - 0.015
        else:
            ha_10 = 'center'
            x_10 = v10

        ax.text(x_5,  yp - 0.09 - 0.16,
                f'\n{v5:.2f}',
                ha=ha_5, va='top', fontsize=7.0, color=c,
                fontweight='500', linespacing=1.3, zorder=5)
        ax.text(x_10, yp + 0.09 + 0.16,
                f'{v10:.2f}',
                ha=ha_10, va='bottom', fontsize=7.0, color=c,
                fontweight='500', linespacing=1.3, zorder=5)
                
        # 100% reference label at right edge
        ax.text(min(ref + 0.016, 1.03), yp,
                f'{ref:.2f}\n(n={tot})',
                ha='left', va='center', fontsize=6.5, color=c,
                alpha=0.52, linespacing=1.3, zorder=5)


def _dotstrip_axes(ax, y_pos, show_yticks, mlabel):
    """Apply shared axis formatting to a dotstrip panel."""
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.65, len(SETS) - 0.35)
    ax.set_yticks(list(y_pos.values()))
    if show_yticks:
        ax.set_yticklabels([LABELS[s] for s in reversed(SETS)], fontsize=8)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(mlabel, fontsize=8.5, labelpad=3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '1.00'], fontsize=7)
    ax.axvline(1.0, color='#e8e8e8', linewidth=0.6, zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines['left'].set_linewidth(0.4)
    ax.tick_params(left=show_yticks)
    for yi in range(len(SETS) - 1):
        ax.axhline(yi + 0.5, color='#f0f0f0', linewidth=0.5, zorder=0)


def _dotstrip_legend(fig):
    leg_items = [
        plt.scatter([], [], s=55,  c='#888780', edgecolors='white',
                    linewidths=0.7,
                    label='5% of gene set  (value below dot)'),
        plt.scatter([], [], s=120, c='#888780', edgecolors='white',
                    linewidths=0.7,
                    label='10% of gene set  (value above dot)'),
        mpatches.Patch(facecolor='#888780', alpha=0.13,
                       label='100% reference bar  (value at right edge)'),
    ]
    fig.legend(handles=leg_items, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=True,
               framealpha=0.95, edgecolor='#ddd',
               fontsize=7, handlelength=1.6, columnspacing=1.0)


def make_dotstrip_single_fig(ser, dataset_label, file_label, out_path):
    """
    Save a dotstrip plot for a single file/resolution.
    It contains 2 panels side-by-side:
      Left panel: PR-AUC at 5% & 10% vs 100% reference
      Right panel: ROC-AUC at 5% & 10% vs 100% reference
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.4, 3.8),
        constrained_layout=True,
    )
    
    psnap = get_pct_snaps(ser)
    metrics = [('pr', 'PR-AUC'), ('auc', 'ROC-AUC')]
    
    y_pos = {s: i for i, s in enumerate(reversed(SETS))}
    def y_fn(s): return float(y_pos[s])
    
    for ci, (mkey, mlabel) in enumerate(metrics):
        ax = axes[ci]
        _draw_dotstrip_rows(ax, psnap, y_fn, mkey)
        _dotstrip_axes(ax, y_pos, ci == 0, mlabel)
        
    _dotstrip_legend(fig)
    fig.suptitle(
        f'{dataset_label} ({file_label})  ·  PR-AUC & ROC-AUC at 5% and 10% of features',
        fontsize=9.5, fontweight='bold', y=1.02,
    )
    fig.savefig(out_path, format='pdf', dpi=300,
                bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)
    print(f'Saved: {out_path}')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def sanitize_filename(name):
    return name.replace('.csv', '').replace(' ', '_').replace('.', '_').replace('-', '_').lower()


def main():
    parser = argparse.ArgumentParser(
        description='Generate gene-set classifier performance figures.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--out', default='.',
        help='Output directory for PDF files (default: current directory)',
    )
    parser.add_argument(
        '--delimiter', default=',',
        help='Column delimiter in input files (default: comma; use \\t for TSV)',
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    delim = args.delimiter.replace('\\t', '\t')

    for ds, res_map in LEIDEN_FILES.items():
        ds_label     = f'PBMC {ds.upper()}'
        leiden_series = {rl: load_series(p, delim)
                         for rl, p in res_map.items()}
        leiden_snaps  = {rl: snap_fixed_n(ser)
                         for rl, ser in leiden_series.items()}

        if ds in ['3k', '10k']:
            prefix = f"pbmc{ds}"
        else:
            prefix = ds

        def op(name):
            return os.path.join(args.out, f'{prefix}_{name}.pdf')

        # Generate separate curve and dot plots for each file
        for rl, ser in leiden_series.items():
            rl_sanitized = sanitize_filename(rl)
            make_curve_single_fig(ser, ds_label, rl, op(f'{rl_sanitized}_curves'))
            make_dotstrip_single_fig(ser, ds_label, rl, op(f'{rl_sanitized}_dotplot'))

        # Cleveland plots
        make_cleveland_plot(leiden_snaps, ds_label, 'auc', op('cleveland_rocauc'))
        make_cleveland_plot(leiden_snaps, ds_label, 'pr', op('cleveland_prauc'))

    print('\nAll figures saved.')


if __name__ == '__main__':
    import argparse
    main()
