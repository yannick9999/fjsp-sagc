"""Temporary, one-off combined IQM plot across no_unpooling / indist / ood.

Reads the already-computed analysis caches (no bootstrap recomputation) and
draws one grouped bar chart with all three experiments side by side, for a
fixed set of sizes. Not part of the regular pipeline -- delete when done.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent

SIZES = ["20x10", "30x10", "40x10", "50x10", "100x10", "edata", "rdata", "vdata"]

EXPERIMENTS = [
    ("no_unpooling", RESULTS_DIR / "no_unpooling" / "analysis_cache.pkl"),
    ("indist", RESULTS_DIR / "multi_size_training" / "analysis_cache_indist.pkl"),
    ("ood", RESULTS_DIR / "multi_size_training" / "analysis_cache_ood.pkl"),
]
EXPERIMENT_LABELS = {"no_unpooling": "NoUnpooling", "indist": "InDist", "ood": "OOD"}
# One color per experiment; greedy/sample distinguished by hatch, sagc/nopooling by shade.
EXPERIMENT_COLORS = {"no_unpooling": "#4C72B0", "indist": "#55A868", "ood": "#C44E52"}
METHOD_SHADE = {"sagc": 1.0, "nopooling": 0.6}  # alpha multiplier
MODE_HATCHES = {"greedy": "", "sample": "///"}
MODE_LABELS = {"greedy": "Greedy", "sample": "Sampling"}

BASELINE_COLORS = {"CPSAT": "#9A7090", "BestDR": "#2E7D6B"}
METHODS = ["sagc"]
MODES = ["greedy", "sample"]


def load_size_entry(cache_path: Path, size: str) -> dict | None:
    with open(cache_path, "rb") as f:
        data = pickle.load(f)
    entry = data["iqm_bars"].get(size) or data["iqm_bars_hurink"].get(size)
    return entry


def main():
    mpl.rcParams.update({
        'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'figure.dpi': 150, 'savefig.dpi': 300,
    })

    combos = [(exp, method, mode) for exp, _ in EXPERIMENTS for mode in MODES for method in METHODS]
    n_sizes = len(SIZES)
    n_bars = len(combos) + 1  # +1 for BestDR bar (CPSAT is a reference line)
    bar_width = 0.18
    group_gap = 0.7
    group_positions = np.arange(n_sizes) * (n_bars * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

    max_top, min_bottom = 1.04, 0.85
    combo_handles = []

    for c_idx, (exp, method, mode) in enumerate(combos):
        offsets = group_positions + c_idx * bar_width
        means, err_low, err_high = [], [], []
        for size in SIZES:
            cache_path = dict(EXPERIMENTS)[exp]
            entry = load_size_entry(cache_path, size)
            key = f"{method}__{mode}"
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

        color = EXPERIMENT_COLORS[exp]
        bars = ax.bar(offsets, means, width=bar_width,
                      color=color, alpha=METHOD_SHADE[method], hatch=MODE_HATCHES[mode],
                      yerr=[err_low, err_high], capsize=1.5,
                      error_kw={"elinewidth": 0.6, "capthick": 0.6},
                      edgecolor="white", linewidth=0.4, zorder=3)
        combo_handles.append(bars[0])

    # BestDR bar (deterministic, no CI), one per size, placed after all combos
    bestdr_offsets = group_positions + len(combos) * bar_width
    bestdr_means = []
    for size in SIZES:
        # BestDR is identical across experiments (same baseline data), use the first available
        val = None
        for exp, _ in EXPERIMENTS:
            entry = load_size_entry(dict(EXPERIMENTS)[exp], size)
            if entry and "BestDR" in entry["baseline_iqm"]:
                val = entry["baseline_iqm"]["BestDR"]
                break
        bestdr_means.append(val if val is not None else np.nan)
        if val is not None:
            max_top = max(max_top, val)
            min_bottom = min(min_bottom, val)

    bestdr_bars = ax.bar(bestdr_offsets, bestdr_means, width=bar_width,
                         color=BASELINE_COLORS["BestDR"], edgecolor="white", linewidth=0.4, zorder=3)

    # CP-SAT reference line at 1.0
    group_centers = group_positions + (n_bars - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(SIZES, fontsize=12)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_bars * bar_width + 0.4)

    max_top = max(max_top, 1.0)
    min_bottom = min(min_bottom, 1.0)
    cpsat_handle = ax.axhline(1.0, color=BASELINE_COLORS["CPSAT"], linewidth=1.8, zorder=4)

    ax.set_ylim(min_bottom - 0.02, max_top + 0.02)
    ax.set_ylabel("IQM Score (C_CP-SAT / C)", fontsize=13, labelpad=8)
    ax.tick_params(axis='y', labelsize=11)

    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend: experiment x mode (color+hatch), sagc/nopooling collapsed via alpha note in label
    legend_handles, legend_labels = [], []
    for exp, _ in EXPERIMENTS:
        for mode in MODES:
            for method in METHODS:
                idx = combos.index((exp, method, mode))
                legend_handles.append(combo_handles[idx])
                legend_labels.append(f"{EXPERIMENT_LABELS[exp]} {method} ({MODE_LABELS[mode]})")
    legend_handles += [bestdr_bars[0], cpsat_handle]
    legend_labels += ["Best DR", "CP-SAT"]

    ax.legend(legend_handles, legend_labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=9,
              frameon=True, framealpha=0.9, edgecolor="#CCCCCC", handlelength=1.5)

    ax.set_title("IQM Score -- NoUnpooling vs. Multi-Size Training (InDist/OOD)",
                 fontsize=15, fontweight='bold', pad=12)

    fig.tight_layout()
    out_path = RESULTS_DIR / "tmp_combined_iqm.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
