"""Loads pilot test result data, runs the rliable bootstrap analysis, and
writes the gap table plus a cache of everything plot.py needs.

Run this whenever the underlying data changes. Run plot.py (no
recomputation) whenever only the plot styling should change.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

from common import (
    ANALYSIS_CACHE,
    BASELINES,
    BENCHMARKS_DIR,
    BOOTSTRAP_REPS,
    MK_SIZE,
    METHOD_LABELS,
    METHODS,
    PLOTS_DIR,
    SCRIPT_DIR,
    SEEDS,
    SIZE_FOLDER_MAP,
    TEST_SIZES,
)


# Data loading

def _find_excel(folder: Path, pattern: str = "*.xlsx"):
    """Finds the newest Excel file in the folder."""
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load_drl_test_makespans(method: str, size: str, seed: int) -> dict[str, float] | None:
    """Loads test makespans per instance for a method, size, seed.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_20x10" / "test" / f"seed{seed}" / f"{folder_size}_greedy"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    # Column 0 is file_name, column 1 is the model checkpoint (makespan)
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_drl_overhead(method: str, size: str, seed: int) -> pd.DataFrame | None:
    """Loads coarsening_overhead sheet."""
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_20x10" / "test" / f"seed{seed}" / f"{folder_size}_greedy"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    try:
        return pd.read_excel(excel, sheet_name="coarsening_overhead")
    except Exception:
        return None


def load_drl_training_curve(method: str, seed: int) -> pd.DataFrame | None:
    """Loads validation_curve for a method and seed."""
    folder = SCRIPT_DIR / f"{method}_20x10" / f"seed{seed}"
    excel = _find_excel(folder, "train_results_*.xlsx")
    if excel is None:
        return None
    return pd.read_excel(excel, sheet_name="validation_curve")


def load_benchmark_makespans(rule: str, size: str) -> dict[str, float] | None:
    """Loads benchmark makespans from CSV.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    csv = BENCHMARKS_DIR / rule / f"{SIZE_FOLDER_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return dict(zip(df["instance_name"].astype(str), df["makespan"].astype(float)))


# Score matrices

def get_baseline_makespans(size: str) -> dict[str, dict[str, float]]:
    """Collects all available baseline makespans for a size.

    Returns:
        Dict {baseline_name: {instance_name: makespan}}
    """
    result = {}
    for b in BASELINES:
        m = load_benchmark_makespans(b, size)
        if m is not None:
            result[b] = m
        else:
            print(f"  [warn] Baseline {b} missing for {size}")
    return result


def compute_c_best(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes C_best per instance as the minimum over all baselines."""
    if not baseline_data:
        return {}
    # Collect all instances
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
    """Builds the normalized score matrix for a method and size.

    Score = C_best / C_drl (higher = better).

    Returns:
        (matrix shape (num_seeds, num_instances), list of instance_names in
         the same order as the matrix columns)
    """
    per_seed_dicts = []
    for s in SEEDS:
        d = load_drl_test_makespans(method, size, s)
        if d is None:
            print(f"  [warn] Test data missing: {method} seed{s} {size}")
            return np.array([]), []
        per_seed_dicts.append(d)

    # Common instances present in all seeds AND in c_best
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
    """Score array for a deterministic baseline (shape (1, num_instances))."""
    scores = np.array([c_best[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


# Bootstrap analysis (the expensive rliable calls; results get cached for plot.py)

def analyze_training_curves() -> dict:
    """Aggregates (mean/min/max) training curves per method across seeds."""
    result = {}
    for method in METHODS:
        curves = []
        for s in SEEDS:
            df = load_drl_training_curve(method, s)
            if df is None:
                print(f"  [warn] Training data missing: {method} seed{s}")
                continue
            curves.append(df)
        if not curves:
            continue

        # Reduce to common env_steps
        min_len = min(len(c) for c in curves)
        env_steps = curves[0]["env_steps"].values[:min_len]
        makespans = np.stack([c["makespan_avg"].values[:min_len] for c in curves])

        result[method] = {
            "env_steps": env_steps,
            "mean": makespans.mean(axis=0),
            "lo": makespans.min(axis=0),
            "hi": makespans.max(axis=0),
        }
    return result


def analyze_iqm_bars(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                     baseline_scores_per_size: dict[str, dict[str, np.ndarray]]) -> dict:
    """Bootstraps IQM + 95% CI per method, per size; plus baseline IQM points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    result = {}
    n = len(TEST_SIZES)

    for si, size in enumerate(TEST_SIZES):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            result[size] = None
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=BOOTSTRAP_REPS)
        print(f"{time.time()-t0:.1f}s")

        methods = list(score_dict.keys())
        means = {m: float(iqm_scores[m][0]) for m in methods}
        cis = {m: (float(iqm_cis[m][0, 0]), float(iqm_cis[m][1, 0])) for m in methods}

        baseline_scores = baseline_scores_per_size.get(size, {})
        baseline_iqm = {}
        for b, arr in baseline_scores.items():
            if arr.size:
                baseline_iqm[b] = float(metrics.aggregate_iqm(arr))

        result[size] = {"methods": methods, "means": means, "cis": cis, "baseline_iqm": baseline_iqm}

    return result


def analyze_performance_profiles(score_dict_per_size: dict[str, dict[str, np.ndarray]]) -> dict:
    """Bootstraps performance profiles (score distribution over tau) per size."""
    tau_list = np.linspace(0.75, 1.05, 50)
    result = {"tau_list": tau_list, "sizes": {}}
    n = len(TEST_SIZES)

    for si, size in enumerate(TEST_SIZES):
        score_dict = score_dict_per_size.get(size, {})
        if not score_dict:
            result["sizes"][size] = None
            continue

        print(f"    bootstrap {size} ({si+1}/{n}) ...", end=" ", flush=True)
        t0 = time.time()
        score_distr, score_distr_cis = rly.create_performance_profile(
            score_dict, tau_list, reps=BOOTSTRAP_REPS
        )
        print(f"{time.time()-t0:.1f}s")

        result["sizes"][size] = {"score_distr": score_distr, "score_distr_cis": score_distr_cis}

    return result


def analyze_probability_of_improvement(score_dict_per_size: dict[str, dict[str, np.ndarray]]) -> dict:
    """Bootstraps P(method1 > method2), one value per instance size."""
    if len(METHODS) < 2:
        return {}
    m1, m2 = METHODS[0], METHODS[1]
    key = f"{m1}_gt_{m2}"

    means, lows, highs, sizes_with_data = [], [], [], []

    for si, size in enumerate(TEST_SIZES):
        sd = score_dict_per_size.get(size, {})
        if m1 not in sd or m2 not in sd:
            continue
        print(f"    bootstrap {size} ({si+1}/{len(TEST_SIZES)}) ...", end=" ", flush=True)
        t0 = time.time()
        pair_dict = {key: (sd[m1], sd[m2])}
        poi, poi_cis = rly.get_interval_estimates(
            pair_dict, metrics.probability_of_improvement, reps=BOOTSTRAP_REPS
        )
        print(f"{time.time()-t0:.1f}s")
        means.append(float(np.squeeze(poi[key])))
        ci = poi_cis[key]
        lows.append(float(np.squeeze(ci[0])))
        highs.append(float(np.squeeze(ci[1])))
        sizes_with_data.append(size)

    return {"m1": m1, "m2": m2, "sizes": sizes_with_data, "means": means, "lows": lows, "highs": highs}


def analyze_scaling(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                    baseline_scores_per_size: dict[str, dict[str, np.ndarray]]) -> dict:
    """Bootstraps IQM vs. instance size, one series per method; plus baseline points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])

    methods_result = {}
    total_bs = len(METHODS) * len(TEST_SIZES)
    bs_i = 0
    for method in METHODS:
        means, lows, highs = [], [], []
        for size in TEST_SIZES:
            sd = score_dict_per_size.get(size, {})
            bs_i += 1
            if method not in sd:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            print(f"    bootstrap {method} {size} ({bs_i}/{total_bs}) ...", end=" ", flush=True)
            t0 = time.time()
            single = {method: sd[method]}
            iqm_scores, iqm_cis = rly.get_interval_estimates(single, iqm_fn, reps=BOOTSTRAP_REPS)
            print(f"{time.time()-t0:.1f}s")
            means.append(float(iqm_scores[method][0]))
            lows.append(float(iqm_cis[method][0, 0]))
            highs.append(float(iqm_cis[method][1, 0]))
        methods_result[method] = {"means": means, "lows": lows, "highs": highs}

    baselines_result = {}
    for b in BASELINES:
        means = []
        for size in TEST_SIZES:
            arr = baseline_scores_per_size.get(size, {}).get(b)
            means.append(np.nan if arr is None or arr.size == 0 else float(metrics.aggregate_iqm(arr)))
        if not all(np.isnan(means)):
            baselines_result[b] = means

    return {"methods": methods_result, "baselines": baselines_result}


def create_gap_table():
    """Table 6: Mean makespan and mean gap per method x instance size.

    Also includes Mk (Brandimarte) at the end. Writes the CSV directly;
    this is a table, not a plot, so it doesn't go through plot.py.
    """
    all_sizes = TEST_SIZES + [MK_SIZE]
    rows = []

    for size in all_sizes:
        baseline_data = get_baseline_makespans(size)
        c_best = compute_c_best(baseline_data)

        # DRL methods
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

    # Also display in terminal
    print()
    print(df.to_string(index=False))


# Main

def main():
    print("=" * 70)
    print("Pilot Test Analysis")
    print("=" * 70)
    print(f"Script dir:    {SCRIPT_DIR}")
    print(f"Benchmarks:    {BENCHMARKS_DIR}")
    print(f"Cache out:     {ANALYSIS_CACHE}")
    print(f"Methods:       {METHODS}")
    print(f"Seeds:         {SEEDS}")
    print(f"Test sizes:    {TEST_SIZES}")
    print(f"Baselines:     {BASELINES}")
    print()

    # 1. Prepare score matrices per size
    print("Loading data and building score matrices ...")
    score_dict_per_size = {}
    baseline_scores_per_size = {}

    for size in TEST_SIZES:
        print(f"  Size {size}")
        baseline_data = get_baseline_makespans(size)
        c_best = compute_c_best(baseline_data)
        if not c_best:
            print(f"  [warn] No C_best for {size}, skipping")
            continue

        score_dict = {}
        for method in METHODS:
            matrix, instances = build_score_matrix(method, size, c_best)
            if matrix.size == 0:
                continue
            score_dict[method] = matrix
            print(f"    {method}: {matrix.shape}, IQM={np.mean(np.sort(matrix.flatten())[len(matrix.flatten())//4:3*len(matrix.flatten())//4]):.4f}")
        score_dict_per_size[size] = score_dict

        # Baselines as (1, num_instances) arrays
        if score_dict:
            # Use the instance list of the first method as reference
            first_method = next(iter(score_dict))
            _, instances = build_score_matrix(first_method, size, c_best)
            baseline_scores = {}
            for b, b_data in baseline_data.items():
                arr = build_baseline_score(b_data, c_best, instances)
                if arr.size:
                    baseline_scores[b] = arr
            baseline_scores_per_size[size] = baseline_scores

    cache = {}
    steps = [
        ("06 Gap Table",                           create_gap_table, None),
        ("Training Curves (aggregate)",            analyze_training_curves, "training_curves"),
        ("IQM Bars (bootstrap)",                   lambda: analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size), "iqm_bars"),
        ("Performance Profiles (bootstrap)",       lambda: analyze_performance_profiles(score_dict_per_size), "performance_profiles"),
        ("Probability of Improvement (bootstrap)", lambda: analyze_probability_of_improvement(score_dict_per_size), "probability_of_improvement"),
        ("Scaling (bootstrap)",                    lambda: analyze_scaling(score_dict_per_size, baseline_scores_per_size), "scaling"),
    ]

    total = len(steps)
    t_start = time.time()

    print()
    for i, (name, fn, cache_key) in enumerate(steps, 1):
        print(f"[{i}/{total}] {name} ...")
        t0 = time.time()
        out = fn()
        elapsed = time.time() - t0
        print(f"         done in {elapsed:.1f}s")
        if cache_key is not None:
            cache[cache_key] = out

    with open(ANALYSIS_CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved analysis cache to {ANALYSIS_CACHE}")

    total_elapsed = time.time() - t_start
    print()
    print(f"All done in {total_elapsed:.1f}s. Run plot.py to (re)generate plots.")


if __name__ == "__main__":
    main()
