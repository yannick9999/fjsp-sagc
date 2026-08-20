"""Renders the 300x30 test plots from the analysis cache written by
analyze.py.

Run analyze.py first (or whenever the underlying data changes). Re-run this
script alone to restyle plots -- it never recomputes the bootstrap CIs.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from rliable import plot_utils

from common import (
    ANALYSIS_CACHE,
    BASELINE_COLORS,
    BASELINE_LABELS,
    DISPATCHING_RULES,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    MODE_HATCHES,
    MODE_LABELS,
    MODE_LINESTYLES,
    MODES,
    PLOTS_DIR,
    TEST_SIZES,
    combo_key,
    split_combo_key,
)

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


def load_analysis() -> dict:
    if not ANALYSIS_CACHE.exists():
        raise FileNotFoundError(f"{ANALYSIS_CACHE} not found - run analyze.py first.")
    with open(ANALYSIS_CACHE, "rb") as f:
        return pickle.load(f)


def plot_iqm_bars(data: dict, sizes: list[str], out_name: str, title: str, figsize: tuple[float, float]):
    """IQM as a grouped bar chart: one bar per (method, mode) with a
    bootstrap 95% CI, plus one point bar per dispatching rule (deterministic,
    no CI). A reference line at 1.0 marks "as good as the best dispatching
    rule on that instance" -- the anchor the scores are normalized against,
    standing in for the missing CP-SAT ground truth at this size.
    """
    combos = [(method, mode) for mode in MODES for method in METHODS]
    baselines = DISPATCHING_RULES

    n_sizes = len(sizes)
    n_items = len(combos) + len(baselines)
    bar_width = 0.18
    group_gap = 0.5
    group_positions = np.arange(n_sizes) * (n_items * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

    combo_bar_handles = []
    max_top = 1.04
    min_bottom = 0.9
    for c_idx, (method, mode) in enumerate(combos):
        key = combo_key(method, mode)
        offsets = group_positions + c_idx * bar_width
        means, err_low, err_high = [], [], []

        for size in sizes:
            entry = data.get(size)
            if entry and key in entry["means"]:
                val = entry["means"][key]
                means.append(val)
                err_low.append(val - entry["cis"][key][0])
                err_high.append(entry["cis"][key][1] - val)
                max_top = max(max_top, entry["cis"][key][1])
                min_bottom = min(min_bottom, entry["cis"][key][0])
            else:
                means.append(0)
                err_low.append(0)
                err_high.append(0)

        bars = ax.bar(offsets, means, width=bar_width,
                      color=METHOD_COLORS[method], hatch=MODE_HATCHES[mode],
                      yerr=[err_low, err_high],
                      capsize=2, error_kw={"elinewidth": 0.8, "capthick": 0.8},
                      edgecolor="white", linewidth=0.6, zorder=3)
        combo_bar_handles.append(bars[0])

    baseline_bar_handles = []
    for b_idx, baseline in enumerate(baselines):
        offsets = group_positions + (len(combos) + b_idx) * bar_width
        means = []
        for size in sizes:
            entry = data.get(size)
            val = entry["baseline_iqm"].get(baseline) if entry else None
            means.append(val if val is not None else np.nan)
            if val is not None:
                max_top = max(max_top, val)
                min_bottom = min(min_bottom, val)

        bars = ax.bar(offsets, means, width=bar_width,
                      color=BASELINE_COLORS.get(baseline, "gray"),
                      edgecolor="white", linewidth=0.6, zorder=3)
        baseline_bar_handles.append(bars[0])

    group_centers = group_positions + (n_items - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sizes, fontsize=13)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_items * bar_width + 0.4)

    # Best-DR reference line at 1.0, drawn after the x-limits are set so it
    # spans the full width of the axes. This is the normalization anchor
    # itself, not a data series -- always exactly 1.0 by construction.
    max_top = max(max_top, 1.0)
    min_bottom = min(min_bottom, 1.0)
    bestdr_handle = ax.axhline(1.0, color=BASELINE_COLORS.get("BestDR", "black"),
                               linewidth=1.8, zorder=4)

    ax.set_ylim(min_bottom - 0.02, max_top + 0.02)
    ax.set_ylabel("IQM Score (C_BestDR / C)", fontsize=15, labelpad=8)
    ax.tick_params(axis='y', labelsize=13)

    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_handles = combo_bar_handles + [bestdr_handle] + baseline_bar_handles
    all_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m, mo in combos] + \
        ["Best DR (ref.)"] + \
        [BASELINE_LABELS.get(b, b) for b in baselines]
    ax.legend(all_handles, all_labels,
              loc="upper right", fontsize=11,
              frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", handlelength=2.0, ncol=2)

    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_performance_profile(data: dict, size: str, out_name: str, title: str, figsize: tuple[float, float]):
    """Performance profile for SAGC (greedy) at a single instance size."""
    tau_list = data["tau_list"]
    entry = data["sizes"].get(size)
    if not entry:
        print(f"  [warn] No performance profile data for {size}, skipping {out_name}")
        return

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

    score_distr = entry["score_distr"]
    colors, linestyles = {}, {}
    for key in score_distr:
        method, mode = split_combo_key(key)
        colors[key] = METHOD_COLORS[method]
        linestyles[key] = MODE_LINESTYLES[mode]

    plot_utils.plot_performance_profiles(
        score_distr, tau_list,
        performance_profile_cis=entry["score_distr_cis"],
        colors=colors,
        linestyles=linestyles,
        xlabel=r"Normalized Score $\tau$ (C_BestDR / C)",
        ylabel=r"Fraction of runs with score $> \tau$",
        labelsize=13,
        ticklabelsize=11,
        ax=ax,
    )

    ax.axvline(1.0, color=BASELINE_COLORS.get("BestDR", "black"), linewidth=1.2,
               linestyle="--", alpha=0.7, zorder=2, label="Best DR (ref.)")

    ax.grid(False)
    ax.grid(True, axis='y', color='#E0E0E0', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.tick_params(axis='both', length=3, width=0.8, labelsize=11)

    legend_handles = [plt.Line2D([0], [0], color=METHOD_COLORS["sagc"], linestyle=MODE_LINESTYLES["greedy"],
                                  lw=2.5, label=f"{METHOD_LABELS['sagc']} ({MODE_LABELS['greedy']})"),
                      plt.Line2D([0], [0], color=BASELINE_COLORS.get("BestDR", "black"), linestyle="--",
                                  lw=1.2, alpha=0.7, label="Best DR (ref.)")]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10,
              frameon=True, framealpha=0.9, edgecolor='#CCCCCC')

    ax.set_title(title, fontsize=14, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def main():
    print("=" * 70)
    print("300x30 Test Plotting")
    print("=" * 70)
    print(f"Cache in:  {ANALYSIS_CACHE}")
    print(f"Plots out: {PLOTS_DIR}")
    print()

    data = load_analysis()

    steps = [
        ("02 IQM Bars",
         lambda: plot_iqm_bars(data["iqm_bars"], TEST_SIZES, "02_iqm_bars.png",
                               "IQM with 95% Bootstrap CIs (300x30)", (8, 7))),
        ("03 Performance Profile",
         lambda: plot_performance_profile(data["performance_profiles"], "300x30",
                                          "03_performance_profile.png",
                                          "Performance Profile with 95% Bootstrap Confidence Band (300x30)", (6, 5))),
    ]

    total = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        fn()

    print()
    print("All plots done.")


if __name__ == "__main__":
    main()
