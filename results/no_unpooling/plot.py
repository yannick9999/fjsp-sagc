"""Renders all pilot test plots from the analysis cache written by analyze.py.

Run analyze.py first (or whenever the underlying data changes). Re-run this
script alone to restyle plots -- it never recomputes the bootstrap CIs.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from common import (
    ANALYSIS_CACHE,
    BASELINE_COLORS,
    BASELINE_LABELS,
    HURINK_DATASETS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    MODE_HATCHES,
    MODE_LABELS,
    MODE_LINESTYLES,
    MODE_MARKERS,
    PLOTS_DIR,
    TEST_SIZES,
    TEXTWIDTH,
    combo_key,
    set_thesis_style,
    split_combo_key,
)

set_thesis_style()

# Sample-then-greedy, used for bar order and for the order of the two mode
# entries within a method's legend column, so every figure lists them alike.
MODE_ORDER = ["sample", "greedy"]


def load_analysis() -> dict:
    if not ANALYSIS_CACHE.exists():
        raise FileNotFoundError(f"{ANALYSIS_CACHE} not found - run analyze.py first.")
    with open(ANALYSIS_CACHE, "rb") as f:
        return pickle.load(f)


def save_fig(fig, stem: str, svg: bool = False) -> None:
    """Save `fig` as both PDF (for the thesis) and PNG (for quick viewing),
    with the "song_fw_" prefix common to all figures from this script.
    With `svg=True` additionally as SVG, with all text kept as real text
    elements (not outlined to paths) so labels, ticks and legend entries stay
    selectable/editable in vector editors."""
    name = f"song_fw_{stem}"
    fig.savefig(PLOTS_DIR / f"{name}.pdf")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300)
    formats = ".pdf / .png"
    if svg:
        with plt.rc_context({"svg.fonttype": "none"}):
            fig.savefig(PLOTS_DIR / f"{name}.svg", bbox_inches="tight", pad_inches=0.05,
                        metadata={"Date": None})
        formats += " / .svg"
    print(f"  Saved {name} {formats}")


def legend_below(fig, columns: list[list]) -> None:
    """Place a legend underneath the figure, one list of entries per column.

    `columns` is a list of columns, each a list of (handle, label) pairs --
    so entries that belong together (e.g. both modes of one method) sit above
    each other in the same column instead of being spread across a row.
    Matplotlib fills a multi-column legend top-to-bottom within a column, so
    the columns just get concatenated; shorter ones are padded with invisible
    entries to keep every column the same height (otherwise matplotlib
    redistributes the entries and the grouping breaks).

    Drawn as a figure legend with an "outside" location so constrained_layout
    reserves room for it below the x-label, instead of it landing on top of
    the axis labels at some hand-tuned offset.
    """
    columns = [c for c in columns if c]
    if not columns:
        return
    nrow = max(len(c) for c in columns)

    handles, labels = [], []
    for col in columns:
        padded = col + [(plt.Line2D([], [], linestyle="none"), "")] * (nrow - len(col))
        handles.extend(h for h, _ in padded)
        labels.extend(lbl for _, lbl in padded)

    fig.legend(handles, labels, loc="outside lower center",
               ncol=len(columns), frameon=False, handlelength=2.0)


def chunked(entries: list, height: int) -> list[list]:
    """Split a flat list of legend entries into columns of at most `height`,
    so a long group (e.g. the baselines) doesn't stretch the legend into one
    tall column."""
    return [entries[i:i + height] for i in range(0, len(entries), height)]


def plot_training_curves(data: dict):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.45))

    legend_columns = []
    for method in METHODS:
        curve = data.get(method)
        if curve is None:
            continue

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        line, = ax.plot(curve["env_steps"], curve["mean"],
                        color=color, label=label, zorder=3)
        ax.fill_between(curve["env_steps"], curve["lo"], curve["hi"],
                        color=color, alpha=0.15, zorder=2)
        legend_columns.append([(line, label)])

    # Axis labels
    ax.set_xlabel(r"Environment steps ($\times 10^6$)", labelpad=8)
    ax.set_ylabel("Validation makespan", labelpad=8)

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

    # Legend below the axis, one method per column
    legend_below(fig, legend_columns)

    save_fig(fig, "01_training_curves")
    plt.close(fig)


def plot_iqm_bars(data: dict, sizes: list[str], out_stem: str, figsize: tuple[float, float],
                  exclude: set[tuple[str, str, str]] | None = None,
                  baseline_keys: list[str] | None = None,
                  legend_ncol: int = 3, svg: bool = False):
    """IQM as a grouped bar chart. One bar per (method, mode), optionally plus
    one bar per entry in `baseline_keys` (e.g. CP-SAT, best dispatching rule).

    Ordered sample-then-greedy, method innermost (e.g. SAGC_s, NoPooling_s,
    SAGC_g, NoPooling_g) so that within each mode the methods sit side by
    side for direct comparison. Baselines (if shown) come last in each group.

    `exclude` is a set of (size, method, mode) triples to omit entirely (bar
    and error bar left out, not just zeroed) -- e.g. sampling runs that used
    a reduced instance count and would otherwise be misleadingly compared
    against the full-instance-count runs at other sizes.

    `legend_ncol` is the width budget for the legend below the axis: with
    room for more columns than there are methods the baselines get a column
    of their own, otherwise they are stacked underneath the method columns.
    """
    exclude = exclude or set()
    combos = [(method, mode) for mode in MODE_ORDER for method in METHODS]
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
    group_gap = 0.5
    group_positions = np.arange(n_sizes) * (n_items * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize)

    # Draw bars
    combo_bar_handles = []
    max_top = 1.04
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

    # Legend below the axis: one column per method, its two modes stacked, so
    # the two bars of the same color read as a pair.
    handle_by_combo = dict(zip(combos, combo_bar_handles))
    legend_columns = [
        [(handle_by_combo[(method, mode)], f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})")
         for mode in MODE_ORDER if (method, mode) in handle_by_combo]
        for method in METHODS
    ]
    baseline_entries = ([(cpsat_handle, "CP-SAT")] if cpsat_handle else []) + \
        list(zip(baseline_bar_handles, [BASELINE_LABELS.get(b, b) for b in baselines]))
    if legend_ncol > len(legend_columns):
        # Room to the right of the method columns -- baselines go there.
        legend_columns.append(baseline_entries)
    else:
        # Narrow figure: a third column would stretch the legend wider than
        # the axes, so the baselines go into extra rows under the methods.
        for i, entry in enumerate(baseline_entries):
            legend_columns[i % len(legend_columns)].append(entry)
    legend_below(fig, legend_columns)

    save_fig(fig, out_stem, svg=svg)
    plt.close(fig)


def plot_scaling(data: dict):
    """Plot 5: IQM vs. instance size, one line per method."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.55))

    x_labels = TEST_SIZES
    x_pos = np.arange(len(x_labels))

    baseline_column = []
    for b, means in data["baselines"].items():
        line, = ax.plot(x_pos, means, linestyle="--", color=BASELINE_COLORS.get(b, "gray"),
                        alpha=0.7, marker="x", label=b, zorder=2)
        baseline_column.append((line, BASELINE_LABELS.get(b, b)))

    entry_by_combo = {}
    for key, d in data["methods"].items():
        method, mode = split_combo_key(key)
        means = np.array(d["means"])
        label = f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})"
        line, = ax.plot(x_pos, means, label=label,
                        color=METHOD_COLORS[method], linestyle=MODE_LINESTYLES[mode],
                        marker=MODE_MARKERS[mode], zorder=3)
        entry_by_combo[(method, mode)] = (line, label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Instance size", labelpad=8)
    ax.set_ylabel("IQM score", labelpad=8)

    ax.grid(True, axis='y', color='#CCCCCC', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # One column per method (both modes stacked), then the baselines in
    # columns of the same height.
    columns = [[entry_by_combo[(m, mode)] for mode in MODE_ORDER if (m, mode) in entry_by_combo]
               for m in METHODS]
    columns = [c for c in columns if c]
    height = max((len(c) for c in columns), default=2)
    legend_below(fig, columns + chunked(baseline_column, height))

    save_fig(fig, "05_scaling")
    plt.close(fig)


def plot_efficiency(data: dict):
    """Plot 7: Runtime efficiency, one line per (method, mode). Isolates the
    per-decision GNN forward-pass time, i.e. just the compute the
    pooling/no-pooling difference actually touches, with the fixed
    per-instance overhead (env setup, I/O, sampling-loop bookkeeping)
    stripped out. Both inference modes are shown, and they measure different
    regimes: greedy runs one graph per forward pass, so the call is dominated
    by fixed per-call overhead and stays flat across sizes for both methods,
    whereas sampling batches num_sample copies of the instance into one pass
    and therefore actually exercises the per-node compute the pooling
    difference lives in. Log-scale y (the range spans about two orders of
    magnitude). The slowdown factor itself is reported in the surrounding
    text. 200x10 and Hurink are excluded (see EFFICIENCY_SIZES /
    analyze_efficiency).
    """
    sizes = data["sizes"]
    x_pos = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.7, TEXTWIDTH * 0.5))

    legend_columns = []
    for method in METHODS:
        column = []
        for mode in MODE_ORDER:
            entry = data["methods"].get(combo_key(method, mode))
            if entry is None:
                continue
            d = entry["forward_ms"]
            mean = np.array(d["mean"])
            lo = np.array(d["lo"])
            hi = np.array(d["hi"])
            color = METHOD_COLORS[method]
            label = f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})"
            line, = ax.plot(x_pos, mean, color=color, marker=MODE_MARKERS[mode],
                            linestyle=MODE_LINESTYLES[mode], label=label, zorder=3)
            ax.fill_between(x_pos, lo, hi, color=color, alpha=0.15, zorder=2)
            column.append((line, label))
        legend_columns.append(column)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes)
    ax.set_xlabel("Instance size", labelpad=8)
    ax.set_ylabel("Forward pass per decision (ms)", labelpad=8)

    ax.grid(True, axis='y', which='both', color='#CCCCCC', linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_below(fig, legend_columns)

    save_fig(fig, "07_efficiency", svg=True)
    plt.close(fig)


def main():
    print("=" * 70)
    print("Pilot Test Plotting")
    print("=" * 70)
    print(f"Cache in:  {ANALYSIS_CACHE}")
    print(f"Plots out: {PLOTS_DIR}")
    print()

    data = load_analysis()

    steps = [
        ("01 Training Curves",
         lambda: plot_training_curves(data["training_curves"])),
        ("02 IQM Bars",
         lambda: plot_iqm_bars(data["iqm_bars"], TEST_SIZES, "02_iqm_bars",
                               (TEXTWIDTH, TEXTWIDTH * 0.45),
                               exclude={("200x10", "sagc", "sample"), ("200x10", "nopooling", "sample")},
                               baseline_keys=["CPSAT", "BestDR"], svg=True)),
        ("02b IQM Bars (Hurink)",
         lambda: plot_iqm_bars(data["iqm_bars_hurink"], HURINK_DATASETS, "02b_iqm_bars_hurink",
                               (TEXTWIDTH * 0.7, TEXTWIDTH * 0.50),
                               baseline_keys=["CPSAT", "BestDR"], legend_ncol=2)),
        ("05 Scaling",
         lambda: plot_scaling(data["scaling"])),
        ("07 Efficiency",
         lambda: plot_efficiency(data["efficiency"])),
        # Same as 02, but without the best dispatching rule bar.
        ("08 IQM Bars (no BestDR)",
         lambda: plot_iqm_bars(data["iqm_bars"], TEST_SIZES, "08_iqm_bars_no_bestdr",
                               (TEXTWIDTH, TEXTWIDTH * 0.45),
                               exclude={("200x10", "sagc", "sample"), ("200x10", "nopooling", "sample")},
                               baseline_keys=["CPSAT"], svg=True)),
    ]

    total = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        fn()

    print()
    print("All plots done.")


if __name__ == "__main__":
    main()
