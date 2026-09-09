"""Renders all multi-size-training plots from the analysis caches written by
analyze.py.

indist and ood are two separate experiments (see common.py docstring): this
script renders each split's plots into its own plots/{split}/ subfolder from
its own analysis_cache_{split}.pkl. The instance-structure ablation is
unrelated and lives in ablation_plot.py.

Run analyze.py first (or whenever the underlying data changes). Re-run this
script alone to restyle plots -- it never recomputes the bootstrap CIs.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from common import (
    BASELINE_COLORS,
    BASELINE_LABELS,
    HURINK_DATASETS,
    JOB_SWEEP_SIZES,
    MACHINE_SWEEP_SIZES,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    MODE_HATCHES,
    MODE_LABELS,
    MODE_LINESTYLES,
    MODE_MARKERS,
    SPLIT_LABELS,
    SPLITS,
    TEXTWIDTH,
    cache_path,
    combo_key,
    plots_dir,
    set_thesis_style,
)

set_thesis_style()


def load_analysis(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run analyze.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_fig(fig, out_dir: Path, stem: str) -> None:
    """Save `fig` as both PDF (for the thesis) and PNG (for quick viewing),
    with the "multi_size_" prefix common to all figures from this script."""
    name = f"multi_size_{stem}"
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    print(f"  Saved {name}.pdf / .png")


def plot_training_curves(data: dict, split: str, out_dir: Path):
    """Training-curve metric is makespan normalized by MWR (lower is better),
    logged during training for whichever split ('indist_norm'/'ood_norm')
    this pipeline run is analyzing -- not the C_best-based IQM score used
    everywhere else.
    """
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.45))

    for method in METHODS:
        curve = data.get(method)
        if curve is None:
            continue

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        ax.plot(curve["env_steps"], curve["mean"],
                color=color, label=label, zorder=3)
        ax.fill_between(curve["env_steps"], curve["lo"], curve["hi"],
                        color=color, alpha=0.15, zorder=2)

    # Axis labels
    ax.set_xlabel(r"Environment steps ($\times 10^6$)", labelpad=8)
    ax.set_ylabel("Validation makespan / MWR", labelpad=8)

    # X-axis: show 0, 1, 2, 3, 4 with "×10⁶" in axis label
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1e6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}"
    ))

    # Align plot to y-axis
    ax.set_xlim(left=0)

    # Horizontal grid lines only
    ax.grid(True, axis='y', color='#CCCCCC', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              loc='upper right')

    save_fig(fig, out_dir, "01_training_curves")
    plt.close(fig)


def plot_iqm_bars(data: dict, sizes: list[str], out_stem: str, figsize: tuple[float, float],
                  out_dir: Path,
                  exclude: set[tuple[str, str, str]] | None = None,
                  baseline_keys: list[str] | None = None,
                  methods: list[str] | None = None):
    """IQM as a grouped bar chart. One bar per (method, mode), optionally plus
    one bar per entry in `baseline_keys` (e.g. CP-SAT, best dispatching rule).

    Ordered sample-then-greedy, method innermost (e.g. SAGC_s, NoPooling_s,
    SAGC_g, NoPooling_g) so that within each mode the methods sit side by
    side for direct comparison. Baselines (if shown) come last in each group.

    `exclude` is a set of (size, method, mode) triples to omit entirely (bar
    and error bar left out, not just zeroed) -- e.g. sampling runs that used
    a reduced instance count and would otherwise be misleadingly compared
    against the full-instance-count runs at other sizes.

    `methods` restricts which of METHODS gets a bar (default: all of them).
    """
    exclude = exclude or set()
    methods = methods or METHODS
    mode_order = ["sample", "greedy"]
    combos = [(method, mode) for mode in mode_order for method in methods]
    baseline_keys = baseline_keys or []
    # CP-SAT is the normalization anchor (score = C_cpsat / C), so its own
    # score is trivially ~1 everywhere -- shown as a reference line instead
    # of a bar, which also makes it obvious when a bar exceeds it (CP-SAT
    # timing out on larger instances without proving optimality).
    show_cpsat_line = "CPSAT" in baseline_keys
    baselines = [b for b in baseline_keys if b != "CPSAT"]

    n_sizes = len(sizes)
    n_items = len(combos) + len(baselines)
    bar_width = 0.18
    group_gap = 0.35
    group_positions = np.arange(n_sizes) * (n_items * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw bars
    combo_bar_handles = []
    max_top = 1.0
    min_bottom = 0.9
    for c_idx, (method, mode) in enumerate(combos):
        key = combo_key(method, mode)
        offsets = group_positions + c_idx * bar_width
        means, err_low, err_high = [], [], []

        for size in sizes:
            entry = data.get(size)
            if (size, method, mode) in exclude:
                means.append(0)
                err_low.append(0)
                err_high.append(0)
            elif entry and key in entry["means"]:
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
                      capsize=0, error_kw={"elinewidth": 1.0, "ecolor": "black"},
                      edgecolor="white", linewidth=0.6, zorder=3)
        combo_bar_handles.append(bars[0])

    # Baseline bars (e.g. CP-SAT, best dispatching rule) -- deterministic,
    # so no CI/error bar. Missing values are NaN so matplotlib simply skips
    # drawing that bar, instead of a misleading zero-height stub.
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

    # X-axis group labels
    group_centers = group_positions + (n_items - 1) * bar_width / 2
    ax.set_xticks(group_centers)
    ax.set_xticklabels(sizes)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_items * bar_width + 0.4)

    # CP-SAT reference line at 1.0, drawn after the x-limits are set so it
    # spans the full width of the axes.
    cpsat_handle = None
    if show_cpsat_line:
        max_top = max(max_top, 1.0)
        min_bottom = min(min_bottom, 1.0)
        cpsat_handle = ax.axhline(1.0, color=BASELINE_COLORS.get("CPSAT", "black"),
                                  linewidth=1.8, zorder=4)

    # Y-axis -- ceiling/floor grow with the data so bars/error bars that
    # exceed 1.0 (as some Hurink combos do) or fall below 0.9 (as some
    # baselines do) stay fully visible instead of clipping.
    ax.set_ylim(min_bottom - 0.02, max_top + 0.01)
    ax.set_ylabel("IQM score", labelpad=8)

    # Grid and spines
    ax.grid(True, axis="y", color="#CCCCCC", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend below the axis
    all_handles = combo_bar_handles + ([cpsat_handle] if cpsat_handle else []) + baseline_bar_handles
    all_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m, mo in combos] + \
        (["CP-SAT"] if cpsat_handle else []) + \
        [BASELINE_LABELS.get(b, b) for b in baselines]
    ax.legend(all_handles, all_labels, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False,
              handlelength=2.0)

    save_fig(fig, out_dir, out_stem)
    plt.close(fig)


def plot_efficiency(data: dict, out_dir: Path):
    """Plot 7: Runtime efficiency (sampling only). Panel 1 is wall-clock
    solve time per instance -- this includes n_decisions x forward-pass time
    PLUS a fixed per-instance overhead (env setup, I/O, sampling-loop
    bookkeeping) that dominates at small sizes and dilutes the pooling
    effect there. Panel 2 isolates the per-decision GNN forward-pass time,
    i.e. just the compute the pooling/no-pooling difference actually
    touches, with the fixed overhead stripped out -- so the same effect
    shows up earlier and larger. Both log-scale y (the range spans about
    two orders of magnitude). 200x10 and Hurink are excluded (see
    EFFICIENCY_SIZES / analyze_efficiency).
    """
    sizes = data["sizes"]
    x_pos = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.7, TEXTWIDTH * 0.5))

    for method in ["sagc", "nopooling"]:
        d = data["methods"][method]["forward_ms"]
        mean = np.array(d["mean"])
        lo = np.array(d["lo"])
        hi = np.array(d["hi"])
        color = METHOD_COLORS[method]
        ax.plot(x_pos, mean, color=color, marker=MODE_MARKERS["sample"],
                linestyle=MODE_LINESTYLES["sample"],
                label=METHOD_LABELS[method], zorder=3)
        ax.fill_between(x_pos, lo, hi, color=color, alpha=0.15, zorder=2)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Instance size", labelpad=8)
    ax.set_ylabel("Forward pass per decision (ms)", labelpad=8)

    ax.grid(True, axis='y', which='both', color='#CCCCCC', linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              loc='upper left')

    save_fig(fig, out_dir, "07_efficiency")
    plt.close(fig)


def plot_split(split: str):
    out_dir = plots_dir(split)
    print("=" * 70)
    print(f"Multi-Size Training Plotting -- split={split} ({SPLIT_LABELS[split]})")
    print("=" * 70)
    print(f"Cache in:  {cache_path(split)}")
    print(f"Plots out: {out_dir}")
    print()

    data = load_analysis(split)

    # Only the two Song+SAGC series (multi-size and 20x10) are shown in the
    # IQM bars -- Song without SAGC (nopooling) is left out here.
    iqm_methods = ["sagc", "sagc_20x10"]

    steps = [
        ("01 Training Curves",
         lambda: plot_training_curves(data["training_curves"], split, out_dir)),
        ("02 IQM Bars (Jobs)",
         lambda: plot_iqm_bars(data["iqm_bars_jobs"], JOB_SWEEP_SIZES, "02_iqm_bars_jobs",
                               (TEXTWIDTH, TEXTWIDTH * 0.45), out_dir,
                               exclude={("200x10", "sagc", "sample"), ("200x10", "nopooling", "sample"),
                                        ("200x10", "sagc_20x10", "sample")},
                               baseline_keys=["CPSAT", "BestDR"], methods=iqm_methods)),
        ("02b IQM Bars (Machines)",
         lambda: plot_iqm_bars(data["iqm_bars_machines"], MACHINE_SWEEP_SIZES, "02b_iqm_bars_machines",
                               (TEXTWIDTH, TEXTWIDTH * 0.45), out_dir,
                               baseline_keys=["CPSAT", "BestDR"], methods=iqm_methods)),
        ("02c IQM Bars (Hurink)",
         lambda: plot_iqm_bars(data["iqm_bars_hurink"], HURINK_DATASETS, "02c_iqm_bars_hurink",
                               (TEXTWIDTH, TEXTWIDTH * 0.45), out_dir,
                               baseline_keys=["CPSAT", "BestDR"], methods=iqm_methods)),
        ("07 Efficiency",
         lambda: plot_efficiency(data["efficiency"], out_dir)),
    ]

    total = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        fn()
    print()


def main():
    for split in SPLITS:
        plot_split(split)
    print("All plots done.")


if __name__ == "__main__":
    main()
