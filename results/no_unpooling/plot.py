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
    BASELINES,
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


def plot_training_curves(data: dict):
    fig, ax = plt.subplots(figsize=(9, 5))

    for method in METHODS:
        curve = data.get(method)
        if curve is None:
            continue

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        ax.plot(curve["env_steps"], curve["mean"],
                color=color, label=label, linewidth=2.5, zorder=3)
        ax.fill_between(curve["env_steps"], curve["lo"], curve["hi"],
                        color=color, alpha=0.15, zorder=2)

    # Axis labels
    ax.set_xlabel("Environment Steps (×10⁶)", fontsize=12, labelpad=8)
    ax.set_ylabel("Validation Makespan\n(avg over 100 instances)", fontsize=12, labelpad=8)
    ax.set_title("Training Curves on 20×10 Validation Set", fontsize=13, fontweight='bold', pad=12)

    # X-axis: show 0, 1, 2, 3, 4 with "×10⁶" in axis label
    ax.xaxis.set_major_locator(mticker.MultipleLocator(1e6))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1e6)}"
    ))

    # Align plot to y-axis
    ax.set_xlim(left=0)

    # Horizontal grid lines only
    ax.grid(True, axis='y', color='#E0E0E0', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Background colors
    ax.set_facecolor('#F9F9F9')
    fig.patch.set_facecolor('white')

    # Legend
    ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              fontsize=11, loc='upper right')

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 01_training_curves.png")


def plot_iqm_bars(data: dict, sizes: list[str], out_name: str, title: str, figsize: tuple[float, float],
                  exclude: set[tuple[str, str, str]] | None = None, show_baselines: bool = False):
    """IQM as a grouped bar chart. One bar per (method, mode), optionally plus
    one bar per dispatching-rule baseline.

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
    baselines = BASELINES if show_baselines else []

    n_sizes = len(sizes)
    n_items = len(combos) + len(baselines)
    bar_width = 0.18
    group_gap = 0.5
    group_positions = np.arange(n_sizes) * (n_items * bar_width + group_gap)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

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

    # Baseline bars (dispatching rules) -- deterministic, so no CI/error bar.
    baseline_bar_handles = []
    for b_idx, baseline in enumerate(baselines):
        offsets = group_positions + (len(combos) + b_idx) * bar_width
        means = []
        for size in sizes:
            entry = data.get(size)
            val = entry["baseline_iqm"].get(baseline) if entry else None
            means.append(val if val is not None else 0)
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
    ax.set_xticklabels(sizes, fontsize=13)
    ax.set_xlim(group_positions[0] - 0.4, group_positions[-1] + n_items * bar_width + 0.4)

    # Y-axis -- ceiling/floor grow with the data so bars/error bars that
    # exceed 1.0 (as some Hurink combos do) or fall below 0.9 (as some
    # baselines do) stay fully visible instead of clipping.
    ax.set_ylim(min_bottom - 0.02, max_top + 0.02)
    ax.set_ylabel("IQM Score (C_best / C_drl)", fontsize=15, labelpad=8)
    ax.tick_params(axis='y', labelsize=13)

    # Grid and spines
    ax.grid(True, axis="y", color="#E0E0E0", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend top right inside plot
    all_handles = combo_bar_handles + baseline_bar_handles
    all_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m, mo in combos] + list(baselines)
    ax.legend(all_handles, all_labels,
              loc="upper right", fontsize=11,
              frameon=True, framealpha=0.9,
              edgecolor="#CCCCCC", handlelength=2.0, ncol=2 if not baselines else 3)

    # Title
    ax.set_title(title, fontsize=16, fontweight='bold', pad=12)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_performance_profiles(data: dict, sizes: list[str], size_labels: dict[str, str],
                              ncols: int, out_name: str, suptitle: str, figsize: tuple[float, float],
                              suptitle_fontsize: int = 18, legend_fontsize: int = 13,
                              suptitle_y: float = 1.08, legend_y: float = 1.0,
                              axlabelsize: int = 13, ticklabelsize: int = 11,
                              paneltitlesize: int = 14):
    """Performance profiles, one panel per instance size/dataset."""
    tau_list = data["tau_list"]
    n = len(sizes)
    nrows = -(-n // ncols)  # ceil division
    fig, axes = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor('white')

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
            labelsize=axlabelsize,
            ticklabelsize=ticklabelsize,
            wrect=5,
            hrect=5,
            ax=ax,
        )
        ax.set_title(size_labels.get(size, size), fontsize=paneltitlesize, fontweight='bold', pad=10)

        # Horizontal grid lines only, matching the other plots
        ax.grid(False)
        ax.grid(True, axis='y', color='#E0E0E0', linewidth=0.8, zorder=1)
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
        ax.tick_params(axis='both', length=3, width=0.8, labelsize=11)
        ax.set_facecolor('#F9F9F9')

        if legend_handles is None:
            legend_handles = []
            for key in score_distr:
                method, mode = split_combo_key(key)
                legend_handles.append(plt.Line2D(
                    [0], [0], color=METHOD_COLORS[method], linestyle=MODE_LINESTYLES[mode],
                    lw=2.5, label=f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})"))

    for idx in range(n, nrows * ncols):
        axes[divmod(idx, ncols)].axis("off")

    fig.suptitle(suptitle, y=suptitle_y, fontsize=suptitle_fontsize, fontweight='bold')
    if legend_handles:
        fig.legend(handles=legend_handles, loc='upper center',
                   bbox_to_anchor=(0.5, legend_y), ncol=len(legend_handles),
                   frameon=True, framealpha=0.9, edgecolor='#CCCCCC', fontsize=legend_fontsize)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_probability_of_improvement(data: dict, out_name: str, title: str,
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

    fig, axes = plt.subplots(1, len(modes_with_data), figsize=(7 * len(modes_with_data), 5), sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor('white')

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

        ax.set_facecolor('#F9F9F9')
        x_pos = np.arange(len(sizes_with_data))

        ax.bar(x_pos, means, yerr=[err_low, err_high], color=METHOD_COLORS[m1],
               hatch=MODE_HATCHES[mode], edgecolor="white", linewidth=0.6,
               capsize=5, error_kw={"elinewidth": 1.0, "capthick": 1.0},
               width=0.6, zorder=3, label=label)
        ax.axhline(0.5, linestyle="--", color="black", alpha=0.5, zorder=2, label="No difference")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sizes_with_data, fontsize=11)
        ax.set_xlabel("Instance Size", fontsize=12, labelpad=8)
        ax.set_ylim(y_lo, y_hi)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(MODE_LABELS[mode], fontsize=13, fontweight='bold', pad=12)

        ax.grid(True, axis='y', color='#E0E0E0', linewidth=0.8, zorder=1)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                  fontsize=10, loc='upper right')

    axes[0].set_ylabel(label, fontsize=12, labelpad=8)
    fig.suptitle(title, fontsize=15, fontweight='bold')

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_name}")


def plot_scaling(data: dict):
    """Plot 5: IQM vs. instance size, one line per method."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F9F9F9')

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
                marker=MODE_MARKERS[mode], linewidth=2.5, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_xlabel("Instance Size", fontsize=12, labelpad=8)
    ax.set_ylabel("IQM Score (C_best / C_drl)", fontsize=12, labelpad=8)
    ax.tick_params(axis='y', labelsize=11)
    ax.set_title("Scaling: Performance over Instance Size", fontsize=13, fontweight='bold', pad=12)

    ax.grid(True, axis='y', color='#E0E0E0', linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
              fontsize=10, loc='center left', bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_scaling.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 05_scaling.png")


def plot_efficiency(data: dict):
    """Plot 7: Runtime efficiency (sampling only). Left panel is wall-clock
    solve time per instance; right panel is the per-decision GNN forward-pass
    time that drives it. Both vs. instance size, one line per method,
    log-scale y since the range spans about two orders of magnitude.
    200x10 and Hurink are excluded (see EFFICIENCY_SIZES / analyze_efficiency).
    """
    sizes = data["sizes"]
    x_pos = np.arange(len(sizes))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor('white')

    panels = [
        (axes[0], "solve_time", "Solve Time per Instance (s)", "Wall-Clock Solve Time"),
        (axes[1], "forward_ms", "Forward Pass per Decision (ms)", "Per-Decision Forward-Pass Time"),
    ]

    for ax, field, ylabel, title in panels:
        ax.set_facecolor('#F9F9F9')
        for method in METHODS:
            d = data["methods"][method][field]
            mean = np.array(d["mean"])
            lo = np.array(d["lo"])
            hi = np.array(d["hi"])
            color = METHOD_COLORS[method]
            ax.plot(x_pos, mean, color=color, marker=MODE_MARKERS["sample"],
                    linestyle=MODE_LINESTYLES["sample"], linewidth=2.5,
                    label=METHOD_LABELS[method], zorder=3)
            ax.fill_between(x_pos, lo, hi, color=color, alpha=0.15, zorder=2)

        ax.set_yscale("log")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sizes, fontsize=11)
        ax.set_xlabel("Instance Size", fontsize=12, labelpad=8)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
        ax.tick_params(axis='y', labelsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

        ax.grid(True, axis='y', which='both', color='#E0E0E0', linewidth=0.6, zorder=1)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=True, framealpha=0.9, edgecolor='#CCCCCC',
                  fontsize=10, loc='upper left')

    fig.suptitle("Runtime Efficiency (Sampling)", fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 07_efficiency.png")


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
         lambda: plot_iqm_bars(data["iqm_bars"], TEST_SIZES, "02_iqm_bars.png",
                               "Interquartile Mean with 95% Bootstrap CIs", (16, 7),
                               exclude={("200x10", "sagc", "sample"), ("200x10", "nopooling", "sample")})),
        ("02b IQM Bars (Hurink)",
         lambda: plot_iqm_bars(data["iqm_bars_hurink"], HURINK_DATASETS, "02b_iqm_bars_hurink.png",
                               "Interquartile Mean with 95% Bootstrap CIs (Hurink)", (11, 7),
                               show_baselines=True)),
        ("03 Performance Profiles",
         lambda: plot_performance_profiles(data["performance_profiles"], TEST_SIZES, {}, 3,
                                           "03_performance_profiles.png",
                                           "Performance Profiles with 95% Bootstrap Confidence Bands", (13, 12))),
        ("03b Performance Profiles (Hurink)",
         lambda: plot_performance_profiles(data["performance_profiles_hurink"], HURINK_DATASETS, HURINK_LABELS, 3,
                                           "03b_performance_profiles_hurink.png",
                                           "Performance Profiles with 95% Bootstrap Confidence Bands (Hurink)", (13, 5))),
        ("03c Performance Profiles (10x5)",
         lambda: plot_performance_profiles(data["performance_profiles"], ["10x5"], {}, 1,
                                           "03c_performance_profiles_10x5.png",
                                           "Performance Profile with 95% Bootstrap Confidence Bands (10x5)", (5, 5),
                                           suptitle_fontsize=11, legend_fontsize=8,
                                           suptitle_y=1.08, legend_y=1.0,
                                           axlabelsize=9, ticklabelsize=8,
                                           paneltitlesize=10)),
        ("04 Probability of Improvement",
         lambda: plot_probability_of_improvement(data["probability_of_improvement"],
                                                  "04_probability_of_improvement.png",
                                                  "Probability of Improvement",
                                                  exclude={("sample", "200x10")})),
        ("04b Probability of Improvement (Hurink)",
         lambda: plot_probability_of_improvement(data["probability_of_improvement_hurink"],
                                                  "04b_probability_of_improvement_hurink.png",
                                                  "Probability of Improvement (Hurink)")),
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
