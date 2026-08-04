"""Renders the instance-structure ablation plot(s) from the cache written by
instance_structure_analyze.py.

Run instance_structure_analyze.py first (or whenever the underlying data
changes). Re-run this script alone to restyle the plot.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from instance_structure_analyze import COMBOS, CORR_LABELS, CORR_VALUES, FLEX_LABELS, FLEX_VALUES, SPLIT_NAME, combo_key
from common import MODE_HATCHES, MODE_LABELS, MODES, cache_path, plots_dir

mpl.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})

# Distinct from METHOD_COLORS/BASELINE_COLORS in common.py -- this ablation
# is SAGC-only, so color here encodes job-correlation instead of method.
CORR_COLORS = {"000": "#4C72B0", "066": "#C44E52"}


def load_analysis() -> dict:
    path = cache_path(SPLIT_NAME)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run instance_structure_analyze.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def plot_iqm_bars(data: dict, out_dir):
    """IQM (relative to the best dispatching rule) vs. flexibility, grouped
    by job-correlation and mode. One group per flexibility value; within each
    group, one bar per (correlation, mode) combo.
    """
    bar_specs = [(c, mode) for c in CORR_VALUES for mode in MODES]
    n_groups = len(FLEX_VALUES)
    n_bars = len(bar_specs)
    bar_width = 0.18
    group_gap = 0.5
    group_positions = np.arange(n_groups) * (n_bars * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

    max_top = 1.04
    min_bottom = 0.9
    bar_handles = []
    for b_idx, (c, mode) in enumerate(bar_specs):
        offsets = group_positions + b_idx * bar_width
        means, err_low, err_high = [], [], []
        for f in FLEX_VALUES:
            key = combo_key(f, c, mode)
            if key in data["means"]:
                val = data["means"][key]
                lo, hi = data["cis"][key]
                means.append(val)
                err_low.append(val - lo)
                err_high.append(hi - val)
                max_top = max(max_top, hi)
                min_bottom = min(min_bottom, lo)
            else:
                means.append(np.nan)
                err_low.append(0)
                err_high.append(0)

        bars = ax.bar(offsets, means, width=bar_width,
                      color=CORR_COLORS[c], hatch=MODE_HATCHES[mode],
                      yerr=[err_low, err_high],
                      capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                      edgecolor="white", linewidth=0.6, zorder=3)
        bar_handles.append(bars[0])

    group_centers = group_positions + (n_bars - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels([FLEX_LABELS[f] for f in FLEX_VALUES], fontsize=13)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_bars * bar_width + 0.4)
    ax.set_xlabel("Flexibility (avg. eligible machines / operation)", fontsize=13, labelpad=8)

    ax.set_ylim(min_bottom - 0.02, max_top + 0.02)
    ax.set_ylabel("IQM Score (Best DR / C_SAGC)", fontsize=13, labelpad=8)
    ax.tick_params(axis='y', labelsize=12)

    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    labels = [f"Job Corr. {CORR_LABELS[c]} ({MODE_LABELS[mode]})" for c, mode in bar_specs]
    ax.legend(bar_handles, labels, loc="upper right", fontsize=10,
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC", handlelength=2.0)

    ax.set_title("Instance-Structure Ablation: Flexibility × Job Correlation (SAGC, 30x10)",
                fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(out_dir / "01_ablation_iqm_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_ablation_iqm_bars.png")


def main():
    out_dir = plots_dir(SPLIT_NAME)
    print("=" * 70)
    print("Instance-Structure Ablation Plotting")
    print("=" * 70)
    print(f"Cache in:  {cache_path(SPLIT_NAME)}")
    print(f"Plots out: {out_dir}")
    print()

    data = load_analysis()
    plot_iqm_bars(data["iqm_bars"], out_dir)

    print()
    print("All plots done.")


if __name__ == "__main__":
    main()
