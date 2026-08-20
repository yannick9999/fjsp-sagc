"""Loads 300x30 test result data, runs the rliable bootstrap analysis, and
writes the gap table plus a cache of everything plot.py needs.

There is no CP-SAT data at this size (too large to solve to a usable
reference within the time limit), so scores are normalized against the best
of the four dispatching rules per instance (C_bestDR / C) instead of
CP-SAT. A score of 1.0 means "as good as the best dispatching rule on that
instance"; the dispatching rules themselves therefore have IQM <= 1.0 by
construction, and SAGC's score shows how far above/below that bar it lands.

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
    DISPATCHING_RULES,
    METHOD_LABELS,
    METHODS,
    MODE_LABELS,
    MODES,
    PLOTS_DIR,
    SCRIPT_DIR,
    SEEDS,
    SIZE_FOLDER_MAP,
    TEST_SIZES,
    combo_key,
)


# Data loading

def _find_excel(folder: Path, pattern: str = "*.xlsx"):
    """Finds the newest Excel file in the folder."""
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load_drl_test_makespans(method: str, size: str, seed: int, mode: str = "greedy") -> dict[str, float] | None:
    """Loads test makespans per instance for a method, size, seed, mode.

    Returns:
        Dict {instance_name: makespan} or None if the file is missing.
    """
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    # Column 0 is file_name, column 1 is the model checkpoint (makespan)
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_benchmark_makespans(rule: str, size: str) -> dict[str, float] | None:
    """Loads benchmark (dispatching rule) makespans from CSV.

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
    """Collects all available dispatching-rule makespans for a size.

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


def compute_c_best_dr(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes the best-dispatching-rule makespan per instance (the
    normalization anchor, standing in for the missing CP-SAT reference)."""
    if not baseline_data:
        return {}
    all_instances = set()
    for b_data in baseline_data.values():
        all_instances.update(b_data.keys())
    c_best = {}
    for inst in all_instances:
        values = [b_data[inst] for b_data in baseline_data.values() if inst in b_data]
        if values:
            c_best[inst] = min(values)
    return c_best


def build_score_matrix(method: str, size: str, c_best_dr: dict[str, float],
                       mode: str = "greedy") -> tuple[np.ndarray, list[str]]:
    """Builds the normalized score matrix for a method, size, mode.

    Score = C_bestDR / C_drl (higher = better; > 1 means beating every
    dispatching rule on that instance).

    Returns:
        (matrix shape (num_seeds, num_instances), list of instance_names in
         the same order as the matrix columns)
    """
    per_seed_dicts = []
    for s in SEEDS:
        d = load_drl_test_makespans(method, size, s, mode)
        if d is None:
            print(f"  [warn] Test data missing: {method} seed{s} {size} {mode}")
            return np.array([]), []
        per_seed_dicts.append(d)

    # Common instances present in all seeds AND in c_best_dr
    common = set(per_seed_dicts[0].keys())
    for d in per_seed_dicts[1:]:
        common &= set(d.keys())
    if c_best_dr:
        common &= set(c_best_dr.keys())
    instances = sorted(common)

    if not instances:
        return np.array([]), []

    matrix = np.zeros((len(SEEDS), len(instances)))
    for i, s in enumerate(SEEDS):
        for j, inst in enumerate(instances):
            matrix[i, j] = c_best_dr[inst] / per_seed_dicts[i][inst]
    return matrix, instances


def build_baseline_score(baseline_makespans: dict[str, float], c_best_dr: dict[str, float],
                         instances: list[str]) -> np.ndarray:
    """Score array for a deterministic dispatching rule (shape (1, num_instances))."""
    scores = np.array([c_best_dr[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


# Bootstrap analysis (the expensive rliable calls; results get cached for plot.py)

def analyze_iqm_bars(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                     baseline_scores_per_size: dict[str, dict[str, np.ndarray]],
                     sizes: list[str]) -> dict:
    """Bootstraps IQM + 95% CI per (method, mode), per size; plus baseline IQM points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    result = {}
    n = len(sizes)

    for si, size in enumerate(sizes):
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


def analyze_performance_profiles(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                                 sizes: list[str]) -> dict:
    """Bootstraps performance profiles (score distribution over tau) per size."""
    tau_list = np.linspace(0.75, 1.15, 50)
    result = {"tau_list": tau_list, "sizes": {}}
    n = len(sizes)

    for si, size in enumerate(sizes):
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


def create_gap_table():
    """Table: Makespan and gap-to-Best-DR for the four dispatching rules and
    SAGC (greedy), at 300x30. Saved as .xlsx, not a plot.
    """
    combo_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m in METHODS for mo in MODES]
    col_methods = ["BestDR"] + DISPATCHING_RULES + combo_labels
    col_tuples = []
    for m in col_methods:
        col_tuples.append((m, "Makespan"))
        if m != "BestDR":
            col_tuples.append((m, "Gap %"))
    columns = pd.MultiIndex.from_tuples(col_tuples)
    table = pd.DataFrame(index=pd.Index(TEST_SIZES, name="size"), columns=columns)

    for size in TEST_SIZES:
        baseline_data = get_baseline_makespans(size)
        c_best_dr = compute_c_best_dr(baseline_data)

        if c_best_dr:
            table.loc[size, ("BestDR", "Makespan")] = np.mean(list(c_best_dr.values()))

        # Dispatching rules
        for rule in DISPATCHING_RULES:
            b_data = baseline_data.get(rule)
            if b_data is None:
                continue
            table.loc[size, (rule, "Makespan")] = np.mean(list(b_data.values()))
            if c_best_dr:
                common = sorted(set(b_data.keys()) & set(c_best_dr.keys()))
                if common:
                    gaps = [(b_data[i] / c_best_dr[i] - 1) * 100 for i in common]
                    table.loc[size, (rule, "Gap %")] = np.mean(gaps)

        # DRL methods x modes, averaged over seeds
        for method in METHODS:
            for mode in MODES:
                label = f"{METHOD_LABELS[method]} ({MODE_LABELS[mode]})"
                per_seed_makespans, per_seed_gaps = [], []
                for s in SEEDS:
                    d = load_drl_test_makespans(method, size, s, mode)
                    if d is None:
                        continue
                    per_seed_makespans.append(np.mean(list(d.values())))
                    if c_best_dr:
                        common = sorted(set(d.keys()) & set(c_best_dr.keys()))
                        if common:
                            gaps = [(d[i] / c_best_dr[i] - 1) * 100 for i in common]
                            per_seed_gaps.append(np.mean(gaps))
                if per_seed_makespans:
                    table.loc[size, (label, "Makespan")] = np.mean(per_seed_makespans)
                if per_seed_gaps:
                    table.loc[size, (label, "Gap %")] = np.mean(per_seed_gaps)

    xlsx_path = PLOTS_DIR / "01_gap_table.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="gap_table")
    print(f"  Saved 01_gap_table.xlsx ({len(table)} rows)")

    print()
    print(table.to_string())


# Main

def main():
    print("=" * 70)
    print("300x30 Test Analysis (SAGC vs. Dispatching Rules)")
    print("=" * 70)
    print(f"Script dir:    {SCRIPT_DIR}")
    print(f"Benchmarks:    {BENCHMARKS_DIR}")
    print(f"Cache out:     {ANALYSIS_CACHE}")
    print(f"Methods:       {METHODS}")
    print(f"Modes:         {MODES}")
    print(f"Seeds:         {SEEDS}")
    print(f"Test sizes:    {TEST_SIZES}")
    print(f"Baselines:     {BASELINES}")
    print()

    print("Loading data and building score matrices ...")
    score_dict_per_size = {}
    baseline_scores_per_size = {}

    for size in TEST_SIZES:
        print(f"  Size {size}")
        baseline_data = get_baseline_makespans(size)
        c_best_dr = compute_c_best_dr(baseline_data)
        if not c_best_dr:
            print(f"  [warn] No dispatching-rule data for {size}, skipping")
            continue

        score_dict = {}
        for method in METHODS:
            for mode in MODES:
                matrix, instances = build_score_matrix(method, size, c_best_dr, mode)
                if matrix.size == 0:
                    continue
                key = combo_key(method, mode)
                score_dict[key] = matrix
                print(f"    {key}: {matrix.shape}, IQM={np.mean(np.sort(matrix.flatten())[len(matrix.flatten())//4:3*len(matrix.flatten())//4]):.4f}")
        score_dict_per_size[size] = score_dict

        if score_dict:
            first_key = next(iter(score_dict))
            first_method, first_mode = first_key.split("__", 1)
            _, instances = build_score_matrix(first_method, size, c_best_dr, first_mode)
            baseline_scores = {}
            for b, b_data in baseline_data.items():
                arr = build_baseline_score(b_data, c_best_dr, instances)
                if arr.size:
                    baseline_scores[b] = arr
            baseline_scores_per_size[size] = baseline_scores

    cache = {}
    steps = [
        ("01 Gap Table",                     create_gap_table, None),
        ("IQM Bars (bootstrap)",             lambda: analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size, TEST_SIZES), "iqm_bars"),
        ("Performance Profiles (bootstrap)", lambda: analyze_performance_profiles(score_dict_per_size, TEST_SIZES), "performance_profiles"),
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
