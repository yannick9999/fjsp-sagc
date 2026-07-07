"""
Pilot Test Analyse-Skript

Erzeugt folgende Outputs im Ordner ./plots/:
  - 01_training_curves.png/.pdf
  - 02_iqm_bars.png/.pdf
  - 03_performance_profiles.png/.pdf
  - 04_probability_of_improvement.png/.pdf
  - 05_scaling.png/.pdf
  - 06_gap_table.csv

Datenstruktur (relativ zum Speicherort dieses Skripts):
  ./{method}_no_unpooling_20x10/seed{s}/train_results_*.xlsx                # Training
  ./{method}_no_unpooling_20x10/test/seed{s}/{size_folder}_greedy/test_*.xlsx  # Test
  ../benchmarks/{rule}/{size_folder}.csv                                    # DR Benchmarks

Pluggable:
  - Neue Methoden in METHODS hinzufuegen
  - Neue Testgroessen in TEST_SIZES hinzufuegen
  - Weitere Baselines in BASELINES hinzufuegen (CP-SAT, GA, ...)

Dependencies:
  pip install pandas numpy matplotlib openpyxl rliable
"""

from __future__ import annotations

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# =============================================================================
# DATA LOADING
# =============================================================================

def _find_excel(folder: Path, pattern: str = "*.xlsx") -> Path | None:
    """Findet die neueste Excel-Datei im Ordner."""
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load_drl_test_makespans(method: str, size: str, seed: int) -> dict[str, float] | None:
    """Laedt Test-Makespans pro Instanz fuer eine Methode, Groesse, Seed.
    
    Returns:
        Dict {instance_name: makespan} oder None falls Datei fehlt.
    """
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_greedy"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    # Spalte 0 ist file_name, Spalte 1 ist der Modell-Checkpoint (Makespan)
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_drl_overhead(method: str, size: str, seed: int) -> pd.DataFrame | None:
    """Laedt coarsening_overhead Sheet."""
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_greedy"
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


def build_score_matrix(method: str, size: str, c_best: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Baut die normalisierte Score-Matrix fuer eine Methode und Groesse.
    
    Score = C_best / C_drl (hoeher = besser).
    
    Returns:
        (matrix shape (num_seeds, num_instances), liste der instance_names in
         derselben Reihenfolge wie die Spalten der Matrix)
    """
    per_seed_dicts = []
    for s in SEEDS:
        d = load_drl_test_makespans(method, size, s)
        if d is None:
            print(f"  [warn] Test-Daten fehlen: {method} seed{s} {size}")
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
    """Plot 2: IQM mit Bootstrap CIs, ein Panel pro Instanzgroesse."""
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

        methods = list(score_dict.keys())
        x_pos = np.arange(len(methods))
        means = [iqm_scores[m][0] for m in methods]
        # cis shape (2, 1): [low, high]
        err_low = [means[i] - iqm_cis[m][0, 0] for i, m in enumerate(methods)]
        err_high = [iqm_cis[m][1, 0] - means[i] for i, m in enumerate(methods)]
        
        bar_width = 0.20
        colors = [METHOD_COLORS[m] for m in methods]
        ax.bar(x_pos, means, yerr=[err_low, err_high], color=colors, capsize=5,
               edgecolor="black", linewidth=0.5, width=bar_width)
        for xi, val in zip(x_pos, means):
            ax.text(xi, val + max(err_high) + 0.002, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
        ax.set_xlim(-0.5, len(methods) - 0.5)
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
        
        colors = {m: METHOD_COLORS[m] for m in score_dict}
        plot_utils.plot_performance_profiles(
            score_distr, tau_list,
            performance_profile_cis=score_distr_cis,
            colors=colors,
            xlabel=r"Normalized Score $\tau$",
            ax=ax,
        )
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
    """Plot 4: Probability of Improvement, ein Wert pro Instanzgroesse."""
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return
    
    # Nur sinnvoll wenn genau 2 Methoden zum Vergleich da sind
    if len(METHODS) < 2:
        return
    m1, m2 = METHODS[0], METHODS[1]
    label = f"P({METHOD_LABELS[m1]} > {METHOD_LABELS[m2]})"
    
    poi_means, poi_lows, poi_highs = [], [], []
    sizes_with_data = []
    
    for si, size in enumerate(TEST_SIZES):
        sd = score_dict_per_size.get(size, {})
        if m1 not in sd or m2 not in sd:
            continue
        print(f"    bootstrap {size} ({si+1}/{len(TEST_SIZES)}) ...", end=" ", flush=True)
        t0 = time.time()
        pair_dict = {label: (sd[m1], sd[m2])}
        poi, poi_cis = rly.get_interval_estimates(
            pair_dict, metrics.probability_of_improvement, reps=BOOTSTRAP_REPS
        )
        print(f"{time.time()-t0:.1f}s")
        poi_means.append(float(np.squeeze(poi[label])))
        ci = poi_cis[label]
        poi_lows.append(float(np.squeeze(ci[0])))
        poi_highs.append(float(np.squeeze(ci[1])))
        sizes_with_data.append(size)
    
    if not sizes_with_data:
        return
    
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = np.arange(len(sizes_with_data))
    means = np.array(poi_means)
    err_low = means - np.array(poi_lows)
    err_high = np.array(poi_highs) - means
    
    ax.bar(x_pos, means, yerr=[err_low, err_high], color=METHOD_COLORS[m1],
           capsize=5, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, linestyle="--", color="black", alpha=0.5, label="No difference")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sizes_with_data)
    ax.set_ylabel(label)
    ax.set_ylim(0, 1)
    ax.set_title("Probability of Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PLOTS_DIR / f"04_probability_of_improvement.{ext}", dpi=150)
    plt.close(fig)
    print("  Saved 04_probability_of_improvement.png/.pdf")


def plot_scaling(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                 baseline_scores_per_size: dict[str, dict[str, np.ndarray]]):
    """Plot 5: IQM vs. Instanzgroesse, eine Linie pro Methode."""
    if not RLIABLE_AVAILABLE:
        print("  [skip] rliable nicht verfuegbar")
        return
    
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x_labels = TEST_SIZES
    x_pos = np.arange(len(x_labels))
    
    # DRL-Methoden
    total_bs = len(METHODS) * len(TEST_SIZES)
    bs_i = 0
    for method in METHODS:
        means, lows, highs = [], [], []
        for size in TEST_SIZES:
            sd = score_dict_per_size.get(size, {})
            if method not in sd:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                bs_i += 1
                continue
            bs_i += 1
            print(f"    bootstrap {method} {size} ({bs_i}/{total_bs}) ...", end=" ", flush=True)
            t0 = time.time()
            single = {method: sd[method]}
            iqm_scores, iqm_cis = rly.get_interval_estimates(single, iqm_fn, reps=BOOTSTRAP_REPS)
            print(f"{time.time()-t0:.1f}s")
            means.append(float(iqm_scores[method][0]))
            lows.append(float(iqm_cis[method][0, 0]))
            highs.append(float(iqm_cis[method][1, 0]))
        means = np.array(means)
        err_low = means - np.array(lows)
        err_high = np.array(highs) - means
        ax.errorbar(x_pos, means, yerr=[err_low, err_high],
                    label=METHOD_LABELS[method], color=METHOD_COLORS[method],
                    marker="o", capsize=4, linewidth=2)
    
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
        ax.plot(x_pos, means, linestyle="--", color=BASELINE_COLORS.get(b, "gray"),
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
    """Tabelle 6: Mean Makespan und Mean Gap pro Methode x Instanzgroesse.
    
    Beinhaltet auch Mk (Brandimarte) am Schluss.
    """
    all_sizes = TEST_SIZES + [MK_SIZE]
    rows = []
    
    for size in all_sizes:
        baseline_data = get_baseline_makespans(size)
        c_best = compute_c_best(baseline_data)
        
        # DRL-Methoden
        for method in METHODS:
            per_seed_gaps = []
            per_seed_makespans = []
            for s in SEEDS:
                d = load_drl_test_makespans(method, size, s)
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
        for method in METHODS:
            matrix, instances = build_score_matrix(method, size, c_best)
            if matrix.size == 0:
                continue
            score_dict[method] = matrix
            print(f"    {method}: {matrix.shape}, IQM={np.mean(np.sort(matrix.flatten())[len(matrix.flatten())//4:3*len(matrix.flatten())//4]):.4f}")
        score_dict_per_size[size] = score_dict
        
        # Baselines als (1, num_instances) Arrays
        if score_dict:
            # Wir nehmen die Instanzen-Liste der ersten Methode als Referenz
            first_method = next(iter(score_dict))
            _, instances = build_score_matrix(first_method, size, c_best)
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