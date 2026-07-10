"""
Pilot Test Analyse-Skript

Erzeugt folgende Outputs im Ordner ./plots/:
  - 01_training_curves.png/.pdf
  - 02_iqm_bars.png/.pdf
  - 03_performance_profiles.png/.pdf
  - 04_probability_of_improvement.png/.pdf
  - 05_scaling.png/.pdf
  - 06_gap_table.csv
  - 07_runtime_scaling.png/.pdf
  - 07_runtime_table.csv

Datenstruktur (relativ zum Speicherort dieses Skripts):
  ./{method}_no_unpooling_20x10/seed{s}/train_results_*.xlsx                        # Training
  ./{method}_no_unpooling_20x10/test/seed{s}/{size_folder}_{mode}/test_*.xlsx       # Test (mode: greedy/sample)
  ../benchmarks/{rule}/{size_folder}.csv                                            # DR Benchmarks

Jede Methode (SAGC, NoPooling) liegt fuer jede Testgroesse sowohl mit Greedy- als
auch mit Sampling-Decoding vor ("Variante" = Methode x Decode-Modus). Alle Plots
vergleichen sowohl die Methoden untereinander als auch Greedy vs. Sampling.

Pluggable:
  - Neue Methoden in METHODS hinzufuegen
  - Neue Testgroessen in TEST_SIZES hinzufuegen
  - Weitere Baselines in BASELINES hinzufuegen (CP-SAT, GA, ...)

Dependencies:
  pip install pandas numpy matplotlib openpyxl rliable
"""

from __future__ import annotations

import itertools
import time
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# rliable
try:
    from rliable import library as rly
    from rliable import metrics
    from rliable import plot_utils
    RLIABLE_AVAILABLE = True
except ImportError:
    RLIABLE_AVAILABLE = False
    warnings.warn("rliable nicht installiert. pip install rliable")


# =============================================================================
# CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent       # results/
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Methoden, Variantenname (Ordnername ohne "_20x10" und ohne ".pt")
METHODS = ["sagc", "nopooling"]
METHOD_LABELS = {"sagc": "SAGC", "nopooling": "NoPooling"}
METHOD_COLORS = {"sagc": "#1f77b4", "nopooling": "#ff7f0e"}

# Decode-Modi, Ordner-Suffix ("{size_folder}_{mode}")
DECODE_MODES = ["greedy", "sample"]
DECODE_LABELS = {"greedy": "Greedy", "sample": "Sampling"}
DECODE_HATCH = {"greedy": "", "sample": "//"}
DECODE_LINESTYLE = {"greedy": "-", "sample": "--"}
DECODE_MARKER = {"greedy": "o", "sample": "s"}
DECODE_LIGHTEN = {"greedy": 0.0, "sample": 0.45}  # 0 = Originalfarbe, 1 = weiss

# Varianten = Kreuzprodukt aus Methode und Decode-Modus
VARIANTS = list(itertools.product(METHODS, DECODE_MODES))


def variant_key(method: str, mode: str) -> str:
    return f"{method}_{mode}"


def variant_label(method: str, mode: str) -> str:
    return f"{METHOD_LABELS[method]} ({DECODE_LABELS[mode]})"


def _lighten(color: str, amount: float) -> tuple[float, float, float]:
    """Mischt eine Farbe mit Weiss (amount=0 -> Originalfarbe, amount=1 -> weiss)."""
    r, g, b = mcolors.to_rgb(color)
    return (1 - (1 - r) * (1 - amount), 1 - (1 - g) * (1 - amount), 1 - (1 - b) * (1 - amount))


VARIANT_COLORS = {
    variant_key(m, mode): _lighten(METHOD_COLORS[m], DECODE_LIGHTEN[mode])
    for m, mode in VARIANTS
}
VARIANT_LABELS = {variant_key(m, mode): variant_label(m, mode) for m, mode in VARIANTS}

# Seeds, hier 3 fuer den Pilot
SEEDS = [0, 1, 2]

# Testgroessen, einmal als "Anzeige" (mit x) und einmal als Ordnername (ohne x)
TEST_SIZES = ["10x5", "20x5", "15x10", "20x10", "30x10", "40x10", "50x10", "100x10", "200x10"]
SIZE_FOLDER_MAP = {
    "10x5": "1005",
    "20x5": "2005",
    "15x10": "1510",
    "20x10": "2010",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
    "Mk": "brandimarte",
}

# Brandimarte separat, nur fuer die Gap-Tabelle am Schluss
MK_SIZE = "Mk"

# Baselines (Dispatching Rules). CP-SAT und GA spaeter ergaenzen.
BASELINES = ["MWR", "SPT", "MOR", "FIFO"]
BASELINE_COLORS = {
    "MWR": "#d62728",    # rot
    "SPT": "#2ca02c",    # gruen
    "MOR": "#9467bd",    # lila
    "FIFO": "#8c564b",   # braun
    "CPSAT": "#222222",   # spaeter
    "GA": "#666666",      # spaeter
}

# rliable Bootstrap Replikationen
BOOTSTRAP_REPS = 100  # 50000 ist Standard, 1000 reicht fuer Pilottest

# Vergleiche fuer Probability-of-Improvement Plot: (key_a, key_b, kurz_label)
# P(a > b) -- SAGC vs. NoPooling, je einmal fuer Greedy und einmal fuer Sampling
POI_COMPARISONS = [
    (variant_key("sagc", "greedy"), variant_key("nopooling", "greedy"), "Greedy"),
    (variant_key("sagc", "sample"), variant_key("nopooling", "sample"), "Sampling"),
]
POI_COMPARISON_COLORS = {
    "Greedy": METHOD_COLORS["sagc"],
    "Sampling": _lighten(METHOD_COLORS["sagc"], 0.45),
}


# =============================================================================
# DATA LOADING
# =============================================================================

def _find_excel(folder: Path, pattern: str = "*.xlsx") -> Path | None:
    """Findet die neueste Excel-Datei im Ordner."""
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load_drl_test_makespans(method: str, size: str, seed: int, mode: str) -> dict[str, float] | None:
    """Laedt Test-Makespans pro Instanz fuer eine Methode, Groesse, Seed, Decode-Modus.

    Returns:
        Dict {instance_name: makespan} oder None falls Datei fehlt.
    """
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    # Spalte 0 ist file_name, Spalte 1 ist der Modell-Checkpoint (Makespan)
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_drl_overhead(method: str, size: str, seed: int, mode: str) -> pd.DataFrame | None:
    """Laedt coarsening_overhead Sheet."""
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    try:
        return pd.read_excel(excel, sheet_name="coarsening_overhead")
    except Exception:
        return None


def load_drl_training_curve(method: str, seed: int) -> pd.DataFrame | None:
    """Laedt validation_curve fuer eine Methode und Seed."""
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / f"seed{seed}"
    excel = _find_excel(folder, "train_results_*.xlsx")
    if excel is None:
        return None
    return pd.read_excel(excel, sheet_name="validation_curve")


def load_benchmark_makespans(rule: str, size: str) -> dict[str, float] | None:
    """Laedt Benchmark-Makespans aus CSV.

    Returns:
        Dict {instance_name: makespan} oder None falls Datei fehlt.
    """
    csv = BENCHMARKS_DIR / rule / f"{SIZE_FOLDER_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return dict(zip(df["instance_name"].astype(str), df["makespan"].astype(float)))


def load_drl_solve_times(method: str, size: str, seed: int, mode: str) -> dict[str, float] | None:
    """Laedt Solve-Time (Sekunden) pro Instanz fuer eine Methode, Groesse, Seed, Decode-Modus."""
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="solve_time")
    instance_col = df.columns[0]
    time_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[time_col].astype(float)))


def load_benchmark_runtimes(rule: str, size: str) -> dict[str, float] | None:
    """Laedt Runtime (Sekunden) pro Instanz aus der Benchmark-CSV."""
    csv = BENCHMARKS_DIR / rule / f"{SIZE_FOLDER_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if "runtime_seconds" not in df.columns:
        return None
    return dict(zip(df["instance_name"].astype(str), df["runtime_seconds"].astype(float)))


# =============================================================================
# ANALYSIS
# =============================================================================

def get_baseline_makespans(size: str) -> dict[str, dict[str, float]]:
    """Sammelt alle verfuegbaren Baseline-Makespans fuer eine Groesse.

    Returns:
        Dict {baseline_name: {instance_name: makespan}}
    """
    result = {}
    for b in BASELINES:
        m = load_benchmark_makespans(b, size)
        if m is not None:
            result[b] = m
        else:
            print(f"  [warn] Baseline {b} fehlt fuer {size}")
    return result


def compute_c_best(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Berechnet C_best pro Instanz als Minimum ueber alle Baselines."""
    if not baseline_data:
        return {}
    # Alle Instanzen sammeln
    all_instances = set()
    for b_data in baseline_data.values():
        all_instances.update(b_data.keys())
    c_best = {}
    for inst in all_instances:
        values = [b_data[inst] for b_data in baseline_data.values() if inst in b_data]
        if values:
            c_best[inst] = min(values)
    return c_best


def build_score_matrix(method: str, size: str, c_best: dict[str, float], mode: str) -> tuple[np.ndarray, list[str]]:
    """Baut die normalisierte Score-Matrix fuer eine Methode, Groesse und Decode-Modus.

    Score = C_best / C_drl (hoeher = besser).

    Returns:
        (matrix shape (num_seeds, num_instances), liste der instance_names in
         derselben Reihenfolge wie die Spalten der Matrix)
    """
    per_seed_dicts = []
    for s in SEEDS:
        d = load_drl_test_makespans(method, size, s, mode)
        if d is None:
            print(f"  [warn] Test-Daten fehlen: {method}/{mode} seed{s} {size}")
            return np.array([]), []
        per_seed_dicts.append(d)

    # Gemeinsame Instanzen, die in allen Seeds UND in c_best vorkommen
    common = set(per_seed_dicts[0].keys())
    for d in per_seed_dicts[1:]:
        common &= set(d.keys())
    if c_best:
        common &= set(c_best.keys())
    instances = sorted(common)

    if not instances:
        return np.array([]), []

    matrix = np.zeros((len(SEEDS), len(instances)))
    for i, s in enumerate(SEEDS):
        for j, inst in enumerate(instances):
            matrix[i, j] = c_best[inst] / per_seed_dicts[i][inst]
    return matrix, instances


def build_baseline_score(baseline_makespans: dict[str, float], c_best: dict[str, float],
                         instances: list[str]) -> np.ndarray:
    """Score-Array fuer eine deterministische Baseline (shape (1, num_instances))."""
    scores = np.array([c_best[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


# =============================================================================
# PLOTS
# =============================================================================

def plot_training_curves():
    """Plot 1: Validation Makespan ueber env_steps, Methoden im Vergleich."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHODS:
        curves = []
        for s in SEEDS:
            df = load_drl_training_curve(method, s)
            if df is None:
                print(f"  [warn] Training-Daten fehlen: {method} seed{s}")
                continue
            curves.append(df)
        if not curves:
            continue

        # Auf gemeinsame env_steps reduzieren
        min_len = min(len(c) for c in curves)
        env_steps = curves[0]["env_steps"].values[:min_len]
        makespans = np.stack([c["makespan_avg"].values[:min_len] for c in curves])

        mean = makespans.mean(axis=0)
        lo = makespans.min(axis=0)
        hi = makespans.max(axis=0)

        color = METHOD_COLORS[method]
        label = METHOD_LABELS[method]
        ax.plot(env_steps, mean, color=color, label=label, linewidth=2)
        ax.fill_between(env_steps, lo, hi, color=color, alpha=0.2)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Validation Makespan (avg over 100 instances)")
    ax.set_title("Training Curves on 20x10 Validation Set")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"01_training_curves.{ext}", dpi=150)
    plt.close(fig)
    print("  Saved 01_training_curves.png/.pdf")


def plot_iqm_bars(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                  baseline_scores_per_size: dict[str, dict[str, np.ndarray]]):
    """Plot 2: IQM mit Bootstrap CIs, ein Panel pro Instanzgroesse.

    Pro Panel ein Balken je Variante (Methode x Decode-Modus).
    """
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return

    n = len(TEST_SIZES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])

    for si, (ax, size) in enumerate(zip(axes, TEST_SIZES)):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            ax.set_title(f"{size} (keine Daten)")
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=BOOTSTRAP_REPS)
        print(f"{time.time()-t0:.1f}s")

        keys = [variant_key(m, mode) for m, mode in VARIANTS if variant_key(m, mode) in score_dict]
        x_pos = np.arange(len(keys))
        means = [iqm_scores[k][0] for k in keys]
        # cis shape (2, 1): [low, high]
        err_low = [means[i] - iqm_cis[k][0, 0] for i, k in enumerate(keys)]
        err_high = [iqm_cis[k][1, 0] - means[i] for i, k in enumerate(keys)]

        bar_width = 0.6
        colors = [VARIANT_COLORS[k] for k in keys]
        hatches = [DECODE_HATCH[k.split("_")[-1]] for k in keys]
        bars = ax.bar(x_pos, means, yerr=[err_low, err_high], color=colors, capsize=5,
                       edgecolor="black", linewidth=0.5, width=bar_width)
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        for xi, val in zip(x_pos, means):
            ax.text(xi, val + max(err_high) + 0.002, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([VARIANT_LABELS[k] for k in keys], rotation=30, ha="right")
        ax.set_xlim(-0.5, len(keys) - 0.5)
        ax.set_title(size)
        ax.grid(True, alpha=0.3, axis="y")

        # Baselines als horizontale Linien
        baseline_scores = baseline_scores_per_size.get(size, {})
        for b, arr in baseline_scores.items():
            iqm_b = metrics.aggregate_iqm(arr) if arr.size else None
            if iqm_b is not None:
                ax.axhline(iqm_b, linestyle="-", color=BASELINE_COLORS.get(b, "gray"),
                           alpha=0.9, linewidth=1.5, label=b)
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0.7, 1.02)

    axes[0].set_ylabel("IQM Score (C_best / C_drl)")
    fig.suptitle("Interquartile Mean with 95% Bootstrap CIs", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"02_iqm_bars.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 02_iqm_bars.png/.pdf")


def plot_performance_profiles(score_dict_per_size: dict[str, dict[str, np.ndarray]]):
    """Plot 3: Performance Profiles, ein Panel pro Instanzgroesse."""
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return

    n = len(TEST_SIZES)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    tau_list = np.linspace(0.75, 1.05, 50)

    for si, (ax, size) in enumerate(zip(axes, TEST_SIZES)):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            ax.set_title(f"{size} (keine Daten)")
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        score_distr, score_distr_cis = rly.create_performance_profile(
            score_dict, tau_list, reps=BOOTSTRAP_REPS
        )
        print(f"{time.time()-t0:.1f}s")

        colors = {k: VARIANT_COLORS[k] for k in score_dict}
        linestyles = {k: DECODE_LINESTYLE[k.split("_")[-1]] for k in score_dict}
        plot_utils.plot_performance_profiles(
            score_distr, tau_list,
            performance_profile_cis=score_distr_cis,
            colors=colors,
            linestyles=linestyles,
            xlabel=r"Normalized Score $\tau$",
            ax=ax,
        )
        # Legenden-Labels lesbar machen (rliable nutzt die Dict-Keys)
        handles, labels = ax.get_legend_handles_labels()
        labels = [VARIANT_LABELS.get(l, l) for l in labels]
        if handles:
            ax.legend(handles, labels, fontsize=7, loc="upper right")
        ax.set_title(size)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"Fraction of runs with score $> \tau$")
    fig.suptitle("Performance Profiles with 95% Bootstrap Confidence Bands", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"03_performance_profiles.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved 03_performance_profiles.png/.pdf")


def plot_probability_of_improvement(score_dict_per_size: dict[str, dict[str, np.ndarray]]):
    """Plot 4: Probability of Improvement, gruppierte Balken pro Instanzgroesse.

    Vergleicht SAGC vs. NoPooling, je einmal fuer Greedy und einmal fuer Sampling.
    """
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return

    # Pro Vergleich: Liste der (size, mean, low, high)
    results = {label: [] for _, _, label in POI_COMPARISONS}
    sizes_with_any_data = []

    for si, size in enumerate(TEST_SIZES):
        sd = score_dict_per_size.get(size, {})
        pair_dict = {}
        for key_a, key_b, label in POI_COMPARISONS:
            if key_a in sd and key_b in sd:
                pair_dict[label] = (sd[key_a], sd[key_b])
        if not pair_dict:
            continue
        print(f"    bootstrap {size} ({si+1}/{len(TEST_SIZES)}) ...", end=" ", flush=True)
        t0 = time.time()
        poi, poi_cis = rly.get_interval_estimates(
            pair_dict, metrics.probability_of_improvement, reps=BOOTSTRAP_REPS
        )
        print(f"{time.time()-t0:.1f}s")
        sizes_with_any_data.append(size)
        for label in results:
            if label in poi:
                mean = float(np.squeeze(poi[label]))
                ci = poi_cis[label]
                results[label].append((size, mean, float(np.squeeze(ci[0])), float(np.squeeze(ci[1]))))

    if not sizes_with_any_data:
        return

    n_comp = len(POI_COMPARISONS)
    fig, ax = plt.subplots(figsize=(1.6 * len(sizes_with_any_data) + 3, 5))
    x_base = np.arange(len(sizes_with_any_data))
    bar_width = 0.8 / n_comp

    for ci, (_, _, label) in enumerate(POI_COMPARISONS):
        entries = {size: (mean, lo, hi) for size, mean, lo, hi in results[label]}
        means, err_low, err_high, xs = [], [], [], []
        for xi, size in enumerate(sizes_with_any_data):
            if size not in entries:
                continue
            mean, lo, hi = entries[size]
            xs.append(xi + (ci - (n_comp - 1) / 2) * bar_width)
            means.append(mean)
            err_low.append(mean - lo)
            err_high.append(hi - mean)
        if not xs:
            continue
        ax.bar(xs, means, yerr=[err_low, err_high], width=bar_width,
               color=POI_COMPARISON_COLORS.get(label, "gray"), capsize=3,
               edgecolor="black", linewidth=0.5, label=label)

    ax.axhline(0.5, linestyle="--", color="black", alpha=0.5, label="No difference")
    ax.set_xticks(x_base)
    ax.set_xticklabels(sizes_with_any_data)
    ax.set_ylabel("P(SAGC > NoPooling)")
    ax.set_ylim(0, 1)
    ax.set_title("Probability of Improvement: SAGC vs. NoPooling")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"04_probability_of_improvement.{ext}", dpi=150)
    plt.close(fig)
    print("  Saved 04_probability_of_improvement.png/.pdf")


def plot_scaling(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                 baseline_scores_per_size: dict[str, dict[str, np.ndarray]]):
    """Plot 5: IQM vs. Instanzgroesse, eine Linie pro Variante (Methode x Decode-Modus)."""
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return

    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])

    fig, ax = plt.subplots(figsize=(8, 5))

    x_labels = TEST_SIZES
    x_pos = np.arange(len(x_labels))

    # DRL-Varianten
    total_bs = len(VARIANTS) * len(TEST_SIZES)
    bs_i = 0
    for method, mode in VARIANTS:
        key = variant_key(method, mode)
        means, lows, highs = [], [], []
        for size in TEST_SIZES:
            sd = score_dict_per_size.get(size, {})
            if key not in sd:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                bs_i += 1
                continue
            bs_i += 1
            print(f"    bootstrap {key} {size} ({bs_i}/{total_bs}) ...", end=" ", flush=True)
            t0 = time.time()
            single = {key: sd[key]}
            iqm_scores, iqm_cis = rly.get_interval_estimates(single, iqm_fn, reps=BOOTSTRAP_REPS)
            print(f"{time.time()-t0:.1f}s")
            means.append(float(iqm_scores[key][0]))
            lows.append(float(iqm_cis[key][0, 0]))
            highs.append(float(iqm_cis[key][1, 0]))
        means = np.array(means)
        err_low = means - np.array(lows)
        err_high = np.array(highs) - means
        ax.errorbar(x_pos, means, yerr=[err_low, err_high],
                    label=VARIANT_LABELS[key], color=VARIANT_COLORS[key],
                    marker=DECODE_MARKER[mode], linestyle=DECODE_LINESTYLE[mode],
                    capsize=4, linewidth=2)

    # Baselines als gestrichelte Linien
    for b in BASELINES:
        means = []
        for size in TEST_SIZES:
            arr = baseline_scores_per_size.get(size, {}).get(b)
            if arr is None or arr.size == 0:
                means.append(np.nan)
                continue
            means.append(metrics.aggregate_iqm(arr))
        if all(np.isnan(means)):
            continue
        ax.plot(x_pos, means, linestyle=":", color=BASELINE_COLORS.get(b, "gray"),
                alpha=0.7, marker="x", label=b)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Instance Size")
    ax.set_ylabel("IQM Score (C_best / C_drl)")
    ax.set_title("Scaling: Performance over Instance Size")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"05_scaling.{ext}", dpi=150)
    plt.close(fig)
    print("  Saved 05_scaling.png/.pdf")


def create_gap_table():
    """Tabelle 6: Mean Makespan und Mean Gap pro Variante (Methode x Decode-Modus) x Instanzgroesse.

    Beinhaltet auch Mk (Brandimarte) am Schluss.
    """
    all_sizes = TEST_SIZES + [MK_SIZE]
    rows = []

    for size in all_sizes:
        baseline_data = get_baseline_makespans(size)
        c_best = compute_c_best(baseline_data)

        # DRL-Varianten
        for method, mode in VARIANTS:
            per_seed_gaps = []
            per_seed_makespans = []
            for s in SEEDS:
                d = load_drl_test_makespans(method, size, s, mode)
                if d is None:
                    continue
                instances = sorted(set(d.keys()) & set(c_best.keys()))
                if not instances:
                    continue
                makespans = np.array([d[i] for i in instances])
                c_bests = np.array([c_best[i] for i in instances])
                gaps = (makespans / c_bests - 1) * 100
                per_seed_gaps.append(gaps.mean())
                per_seed_makespans.append(makespans.mean())
            if per_seed_gaps:
                rows.append({
                    "size": size,
                    "method": METHOD_LABELS[method],
                    "decode_mode": DECODE_LABELS[mode],
                    "mean_makespan": f"{np.mean(per_seed_makespans):.1f} +/- {np.std(per_seed_makespans):.1f}",
                    "mean_gap_pct": f"{np.mean(per_seed_gaps):.2f} +/- {np.std(per_seed_gaps):.2f}",
                })

        # Baselines
        for b in BASELINES:
            b_data = baseline_data.get(b)
            if b_data is None:
                continue
            instances = sorted(set(b_data.keys()) & set(c_best.keys()))
            if not instances:
                continue
            makespans = np.array([b_data[i] for i in instances])
            c_bests = np.array([c_best[i] for i in instances])
            gaps = (makespans / c_bests - 1) * 100
            rows.append({
                "size": size,
                "method": b,
                "decode_mode": "-",
                "mean_makespan": f"{makespans.mean():.1f}",
                "mean_gap_pct": f"{gaps.mean():.2f}",
            })

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR / "06_gap_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved 06_gap_table.csv ({len(df)} rows)")

    # Auch im Terminal anzeigen
    print()
    print(df.to_string(index=False))


def create_runtime_table_and_plot():
    """Tabelle/Plot 7: Mittlere Solve-Time (Sekunden) pro Instanz, Variante x Instanzgroesse.

    Solve-Time = Zeit fuer die Loesung EINER Instanz (bei Sampling inkl. aller
    Samples). Baselines (Dispatching-Regeln) sind praktisch instantan bei
    kleinen, aber messbar bei grossen Instanzen.
    """
    rows = []
    # Fuer den Plot: Mittelwert pro Groesse x Variante/Baseline
    runtime_means = {key: [] for key in VARIANT_LABELS}  # key -> list aligned to TEST_SIZES
    baseline_means = {b: [] for b in BASELINES}

    for size in TEST_SIZES:
        # DRL-Varianten
        for method, mode in VARIANTS:
            key = variant_key(method, mode)
            per_seed_means = []
            for s in SEEDS:
                d = load_drl_solve_times(method, size, s, mode)
                if d is None:
                    continue
                per_seed_means.append(np.mean(list(d.values())))
            if per_seed_means:
                mean_val = float(np.mean(per_seed_means))
                std_val = float(np.std(per_seed_means))
                rows.append({
                    "size": size,
                    "method": METHOD_LABELS[method],
                    "decode_mode": DECODE_LABELS[mode],
                    "mean_solve_time_sec": f"{mean_val:.3f} +/- {std_val:.3f}",
                })
                runtime_means[key].append(mean_val)
            else:
                runtime_means[key].append(np.nan)

        # Baselines
        for b in BASELINES:
            d = load_benchmark_runtimes(b, size)
            if d is None:
                baseline_means[b].append(np.nan)
                continue
            values = np.array(list(d.values()))
            rows.append({
                "size": size,
                "method": b,
                "decode_mode": "-",
                "mean_solve_time_sec": f"{values.mean():.3f} +/- {values.std():.3f}",
            })
            baseline_means[b].append(values.mean())

    df = pd.DataFrame(rows)
    csv_path = PLOTS_DIR / "07_runtime_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved 07_runtime_table.csv ({len(df)} rows)")
    print()
    print(df.to_string(index=False))

    # Plot: Solve-Time (log) vs. Instanzgroesse
    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = np.arange(len(TEST_SIZES))

    for method, mode in VARIANTS:
        key = variant_key(method, mode)
        means = np.array(runtime_means[key])
        if np.all(np.isnan(means)):
            continue
        ax.plot(x_pos, means, color=VARIANT_COLORS[key], label=VARIANT_LABELS[key],
                marker=DECODE_MARKER[mode], linestyle=DECODE_LINESTYLE[mode], linewidth=2)

    for b in BASELINES:
        means = np.array(baseline_means[b])
        if np.all(np.isnan(means)):
            continue
        ax.plot(x_pos, means, color=BASELINE_COLORS.get(b, "gray"), label=b,
                marker="x", linestyle=":", alpha=0.7)

    ax.set_yscale("log")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(TEST_SIZES)
    ax.set_xlabel("Instance Size")
    ax.set_ylabel("Mean Solve Time per Instance (s, log scale)")
    ax.set_title("Runtime Scaling")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"07_runtime_scaling.{ext}", dpi=150)
    plt.close(fig)
    print("  Saved 07_runtime_scaling.png/.pdf")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Pilot Test Analysis")
    print("=" * 70)
    print(f"Script dir:    {SCRIPT_DIR}")
    print(f"Results dir:   {RESULTS_DIR}")
    print(f"Benchmarks:    {BENCHMARKS_DIR}")
    print(f"Plots out:     {PLOTS_DIR}")
    print(f"Methods:       {METHODS}")
    print(f"Decode modes:  {DECODE_MODES}")
    print(f"Seeds:         {SEEDS}")
    print(f"Test sizes:    {TEST_SIZES}")
    print(f"Baselines:     {BASELINES}")
    print()

    # 1. Score-Matrizen pro Groesse vorbereiten
    print("Loading data and building score matrices ...")
    score_dict_per_size = {}
    baseline_scores_per_size = {}

    for size in TEST_SIZES:
        print(f"  Size {size}")
        baseline_data = get_baseline_makespans(size)
        c_best = compute_c_best(baseline_data)
        if not c_best:
            print(f"  [warn] Kein C_best fuer {size}, ueberspringe")
            continue

        score_dict = {}
        for method, mode in VARIANTS:
            matrix, instances = build_score_matrix(method, size, c_best, mode)
            if matrix.size == 0:
                continue
            key = variant_key(method, mode)
            score_dict[key] = matrix
            print(f"    {key}: {matrix.shape}, IQM={np.mean(np.sort(matrix.flatten())[len(matrix.flatten())//4:3*len(matrix.flatten())//4]):.4f}")
        score_dict_per_size[size] = score_dict

        # Baselines als (1, num_instances) Arrays
        if score_dict:
            # Wir nehmen die Instanzen-Liste der ersten Variante als Referenz
            first_method, first_mode = VARIANTS[0]
            _, instances = build_score_matrix(first_method, size, c_best, first_mode)
            baseline_scores = {}
            for b, b_data in baseline_data.items():
                arr = build_baseline_score(b_data, c_best, instances)
                if arr.size:
                    baseline_scores[b] = arr
            baseline_scores_per_size[size] = baseline_scores

    steps = [
        ("01 Training Curves",            lambda: plot_training_curves()),
        ("02 IQM Bars (bootstrap)",       lambda: plot_iqm_bars(score_dict_per_size, baseline_scores_per_size)),
        ("03 Performance Profiles (bootstrap)", lambda: plot_performance_profiles(score_dict_per_size)),
        ("04 Probability of Improvement (bootstrap)", lambda: plot_probability_of_improvement(score_dict_per_size)),
        ("05 Scaling (bootstrap)",        lambda: plot_scaling(score_dict_per_size, baseline_scores_per_size)),
        ("06 Gap Table",                  lambda: create_gap_table()),
        ("07 Runtime Table + Plot",       lambda: create_runtime_table_and_plot()),
    ]

    total = len(steps)
    t_start = time.time()

    print()
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        t0 = time.time()
        fn()
        elapsed = time.time() - t0
        print(f"         done in {elapsed:.1f}s")

    total_elapsed = time.time() - t_start
    print()
    print(f"All done in {total_elapsed:.1f}s.")


if __name__ == "__main__":
    main()
