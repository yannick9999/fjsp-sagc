"""Renders all pilot test plots from the analysis cache written by analyze.py.

Run analyze.py first (or whenever the underlying data changes). Re-run this
script alone to restyle plots -- it never recomputes the bootstrap CIs.
"""

from __future__ import annotations

import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rliable import plot_utils

from common import (
    ANALYSIS_CACHE,
    BASELINE_COLORS,
    BASELINE_LABELS,
    HURINK_DATASETS,
    HURINK_LABELS,
    METHOD_COLORS,
    METHOD_LABELS,
    METHODS,
    MODE_HATCHES,
    MODE_LABELS,
    MODE_LINESTYLES,
    MODE_MARKERS,
    MODES,
    PLOTS_DIR,
    TEST_SIZES,
    TEXTWIDTH,
    combo_key,
    set_thesis_style,
    split_combo_key,
)

set_thesis_style()


def load_analysis() -> dict:
    if not ANALYSIS_CACHE.exists():
        raise FileNotFoundError(f"{ANALYSIS_CACHE} not found - run analyze.py first.")
    with open(ANALYSIS_CACHE, "rb") as f:
        return pickle.load(f)


def save_fig(fig, stem: str) -> None:
    """Save `fig` as both PDF (for the thesis) and PNG (for quick viewing),
    with the "song_fw_" prefix common to all figures from this script."""
    name = f"song_fw_{stem}"
    fig.savefig(PLOTS_DIR / f"{name}.pdf")
    fig.savefig(PLOTS_DIR / f"{name}.png", dpi=300)
    print(f"  Saved {name}.pdf / .png")


def plot_training_curves(data: dict):
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

    # Legend
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              loc='upper right')

    save_fig(fig, "01_training_curves")
    plt.close(fig)


def plot_iqm_bars(data: dict, sizes: list[str], out_stem: str, figsize: tuple[float, float],
                  exclude: set[tuple[str, str, str]] | None = None,
                  baseline_keys: list[str] | None = None,
                  legend_ncol: int = 3):
    """IQM as a grouped bar chart. One bar per (method, mode), optionally plus
    one bar per entry in `baseline_keys` (e.g. CP-SAT, best dispatching rule).

    Ordered sample-then-greedy, method innermost (e.g. SAGC_s, NoPooling_s,
    SAGC_g, NoPooling_g) so that within each mode the methods sit side by
    side for direct comparison. Baselines (if shown) come last in each group.

    `exclude` is a set of (size, method, mode) triples to omit entirely (bar
    and error bar left out, not just zeroed) -- e.g. sampling runs that used
    a reduced instance count and would otherwise be misleadingly compared
    against the full-instance-count runs at other sizes.
    """
    exclude = exclude or set()
    mode_order = ["sample", "greedy"]
    combos = [(method, mode) for mode in mode_order for method in METHODS]
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

    # Legend below the axis
    all_handles = combo_bar_handles + ([cpsat_handle] if cpsat_handle else []) + baseline_bar_handles
    all_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m, mo in combos] + \
        (["CP-SAT"] if cpsat_handle else []) + \
        [BASELINE_LABELS.get(b, b) for b in baselines]
    ax.legend(all_handles, all_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.12),
              ncol=legend_ncol, frameon=False, handlelength=2.0)

    save_fig(fig, out_stem)
    plt.close(fig)


def plot_performance_profiles(data: dict, sizes: list[str], size_labels: dict[str, str],
                              ncols: int, out_stem: str, figsize: tuple[float, float]):
    """Performance profiles, one panel per instance size/dataset."""
    tau_list = data["tau_list"]
    n = len(sizes)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
    axes = np.atleast_2d(axes)

    # rliable draws its own xlabel/ylabel on every panel at a fixed 'x-large'
    # size; on a multi-row grid that overlaps neighboring panels, so only the
    # bottom-most panel per column (last row may be partial) and the leftmost
    # column get a real label -- everything else gets an empty one.
    bottom_row = {}
    for idx in range(n):
        row, col = divmod(idx, ncols)
        bottom_row[col] = row

    legend_handles = None
    for idx, size in enumerate(sizes):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        entry = data["sizes"].get(size)
        if not entry:
            ax.set_title(f"{size_labels.get(size, size)} (no data)")
            continue

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
            xlabel=r"Normalized Score $\tau$" if row == bottom_row[col] else "",
            ylabel=r"Fraction of runs with score $> \tau$" if col == 0 else "",
            labelsize=mpl.rcParams["axes.labelsize"],
            ticklabelsize=mpl.rcParams["xtick.labelsize"],
            wrect=5,
            hrect=5,
            ax=ax,
        )
        ax.set_title(size_labels.get(size, size), pad=10)

        # Horizontal grid lines only, matching the other plots
        ax.grid(False)
        ax.grid(True, axis='y', color='#CCCCCC', linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)

        # Remove top and right spines; rliable leaves left/bottom thick and
        # pushed outward, which reads as a heavy black bar at this figsize --
        # thin them back down and pull them back to the axis.
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['left'].set_position(('outward', 0))
        ax.spines['bottom'].set_position(('outward', 0))
        ax.tick_params(axis='both', length=3, width=0.8)

        if legend_handles is None:
            legend_handles = []
            for key in score_distr:
                method, mode = split_combo_key(key)
                legend_handles.append(plt.Line2D(
                    [0], [0], color=METHOD_COLORS[method], linestyle=MODE_LINESTYLES[mode],
                    lw=2.5, label=f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})"))

    for idx in range(n, nrows * ncols):
        axes[divmod(idx, ncols)].axis("off")

    if legend_handles:
        fig.legend(handles=legend_handles, loc='outside upper center',
                   ncol=2, frameon=True, framealpha=0.9, edgecolor='#CCCCCC')

    save_fig(fig, out_stem)
    plt.close(fig)


def plot_probability_of_improvement(data: dict, out_stem: str,
                                    exclude: set[tuple[str, str]] | None = None):
    """Probability of improvement, one panel per mode, one bar per instance size.

    `exclude` is a set of (mode, size) pairs to drop before plotting -- e.g.
    sampling runs at a size that used a reduced instance count and would
    otherwise be misleadingly compared against the other sizes/modes.
    """
    exclude = exclude or set()
    modes_with_data = [mo for mo in MODES if data.get(mo) and data[mo].get("sizes")]
    if not modes_with_data:
        return

    fig, axes = plt.subplots(1, len(modes_with_data), figsize=(TEXTWIDTH, TEXTWIDTH * 0.40), sharey=True)
    axes = np.atleast_1d(axes)

    # Filter excluded (mode, size) pairs and track the overall y-range so the
    # shared y-axis can be sized to fit every remaining error bar.
    filtered = {}
    y_min, y_max = 0.5, 0.5
    for mode in modes_with_data:
        d = data[mode]
        sizes_with_data, means, lows, highs = [], [], [], []
        for size, mean, lo, hi in zip(d["sizes"], d["means"], d["lows"], d["highs"]):
            if (mode, size) in exclude:
                continue
            sizes_with_data.append(size)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
        filtered[mode] = {"m1": d["m1"], "m2": d["m2"], "sizes": sizes_with_data,
                          "means": means, "lows": lows, "highs": highs}
        if lows:
            y_min = min(y_min, min(lows))
            y_max = max(y_max, max(highs))

    y_pad = 0.05 * (y_max - y_min)
    y_lo, y_hi = y_min - y_pad, y_max + y_pad

    label = None
    for ax, mode in zip(axes, modes_with_data):
        d = filtered[mode]
        m1, m2 = d["m1"], d["m2"]
        label = f"P({METHOD_LABELS[m1]} > {METHOD_LABELS[m2]})"
        sizes_with_data = d["sizes"]

        means = np.array(d["means"])
        err_low = means - np.array(d["lows"])
        err_high = np.array(d["highs"]) - means

        x_pos = np.arange(len(sizes_with_data))

        ax.bar(x_pos, means, yerr=[err_low, err_high], color=METHOD_COLORS[m1],
               hatch=MODE_HATCHES[mode], edgecolor="white", linewidth=0.6,
               capsize=0, error_kw={"elinewidth": 1.0, "ecolor": "black"},
               width=0.6, zorder=3, label=label)
        ax.axhline(0.5, linestyle="--", color="black", alpha=0.5, zorder=2, label="No difference")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sizes_with_data)
        ax.set_xlabel("Instance size", labelpad=8)
        ax.set_ylim(y_lo, y_hi)
        ax.set_title(MODE_LABELS[mode], pad=12)

        ax.grid(True, axis='y', color='#CCCCCC', linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                  loc='upper right')

    axes[0].set_ylabel(label, labelpad=8)

    save_fig(fig, out_stem)
    plt.close(fig)


def plot_scaling(data: dict):
    """Plot 5: IQM vs. instance size, one line per method."""
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, TEXTWIDTH * 0.55))

    x_labels = TEST_SIZES
    x_pos = np.arange(len(x_labels))

    for b, means in data["baselines"].items():
        ax.plot(x_pos, means, linestyle="--", color=BASELINE_COLORS.get(b, "gray"),
                alpha=0.7, marker="x", label=b, zorder=2)

    for key, d in data["methods"].items():
        method, mode = split_combo_key(key)
        means = np.array(d["means"])
        ax.plot(x_pos, means,
                label=f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})",
                color=METHOD_COLORS[method], linestyle=MODE_LINESTYLES[mode],
                marker=MODE_MARKERS[mode], zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Instance size", labelpad=8)
    ax.set_ylabel("IQM score", labelpad=8)

    ax.grid(True, axis='y', color='#CCCCCC', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              ncol=4, frameon=False)

    save_fig(fig, "05_scaling")
    plt.close(fig)


def plot_efficiency(data: dict):
    """Plot 7: Runtime efficiency (sampling only). Isolates the per-decision
    GNN forward-pass time, i.e. just the compute the pooling/no-pooling
    difference actually touches, with the fixed per-instance overhead (env
    setup, I/O, sampling-loop bookkeeping) stripped out. Log-scale y (the
    range spans about two orders of magnitude). The slowdown factor itself is
    reported in the surrounding text. 200x10 and Hurink are excluded (see
    EFFICIENCY_SIZES / analyze_efficiency).
    """
    sizes = data["sizes"]
    x_pos = np.arange(len(sizes))

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.7, TEXTWIDTH * 0.5))

    for method in METHODS:
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

    save_fig(fig, "07_efficiency")
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
                               baseline_keys=["CPSAT", "BestDR"])),
        ("02b IQM Bars (Hurink)",
         lambda: plot_iqm_bars(data["iqm_bars_hurink"], HURINK_DATASETS, "02b_iqm_bars_hurink",
                               (TEXTWIDTH * 0.7, TEXTWIDTH * 0.50),
                               baseline_keys=["CPSAT", "BestDR"], legend_ncol=2)),
        ("03 Performance Profiles",
         lambda: plot_performance_profiles(data["performance_profiles"], TEST_SIZES, {}, 3,
                                           "03_performance_profiles",
                                           (TEXTWIDTH, TEXTWIDTH * 0.95))),
        ("03b Performance Profiles (Hurink)",
         lambda: plot_performance_profiles(data["performance_profiles_hurink"], HURINK_DATASETS, HURINK_LABELS, 3,
                                           "03b_performance_profiles_hurink",
                                           (TEXTWIDTH, TEXTWIDTH * 0.40))),
        ("03c Performance Profiles (10x5)",
         lambda: plot_performance_profiles(data["performance_profiles"], ["10x5"], {}, 1,
                                           "03c_performance_profiles_10x5",
                                           (TEXTWIDTH * 0.6, TEXTWIDTH * 0.6))),
        ("04 Probability of Improvement",
         lambda: plot_probability_of_improvement(data["probability_of_improvement"],
                                                  "04_probability_of_improvement",
                                                  exclude={("sample", "200x10")})),
        ("04b Probability of Improvement (Hurink)",
         lambda: plot_probability_of_improvement(data["probability_of_improvement_hurink"],
                                                  "04b_probability_of_improvement_hurink")),
        ("05 Scaling",
         lambda: plot_scaling(data["scaling"])),
        ("07 Efficiency",
         lambda: plot_efficiency(data["efficiency"])),
    ]

    total = len(steps)
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        fn()

    print()
    print("All plots done.")


if __name__ == "__main__":
    main()
