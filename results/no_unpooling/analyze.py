"""Loads no_unpooling test result data, runs the rliable bootstrap analysis, and
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
    DISPATCHING_RULES,
    EFFICIENCY_SIZES,
    HURINK_DATASETS,
    MK_SIZE,
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
    split_combo_key,
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
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    # Column 0 is file_name, column 1 is the model checkpoint (makespan)
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_drl_overhead(method: str, size: str, seed: int, mode: str = "greedy") -> pd.DataFrame | None:
    """Loads coarsening_overhead sheet."""
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
    """Loads validation_curve for a method and seed."""
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / f"seed{seed}"
    excel = _find_excel(folder, "train_results_*.xlsx")
    if excel is None:
        return None
    return pd.read_excel(excel, sheet_name="validation_curve")


def load_drl_solve_times(method: str, size: str, seed: int, mode: str = "greedy") -> np.ndarray | None:
    """Loads per-instance solve_time (wall-clock seconds)."""
    folder_size = SIZE_FOLDER_MAP[size]
    folder = SCRIPT_DIR / f"{method}_no_unpooling_20x10" / "test" / f"seed{seed}" / f"{folder_size}_{mode}"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="solve_time")
    return df.iloc[:, 1].astype(float).values


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


def load_cpsat_status(size: str) -> dict[str, str] | None:
    """Loads the CP-SAT solver status (OPTIMAL/FEASIBLE) per instance.

    Returns None if the CSV is missing or doesn't have a status column yet
    (e.g. Mk/Hurink, where CP-SAT data hasn't been backfilled with status).
    """
    csv = BENCHMARKS_DIR / "CPSAT" / f"{SIZE_FOLDER_MAP[size]}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if "status" not in df.columns:
        return None
    return dict(zip(df["instance_name"].astype(str), df["status"].astype(str)))


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


def compute_c_best_dr(baseline_data: dict[str, dict[str, float]]) -> dict[str, float]:
    """Computes the best-dispatching-rule makespan per instance (CP-SAT excluded)."""
    dr_data = {b: d for b, d in baseline_data.items() if b in DISPATCHING_RULES}
    return compute_c_best(dr_data)


def build_score_matrix(method: str, size: str, c_cpsat: dict[str, float],
                       mode: str = "greedy") -> tuple[np.ndarray, list[str]]:
    """Builds the normalized score matrix for a method, size, mode.

    Score = C_cpsat / C_drl (higher = better; > 1 means beating CP-SAT,
    which happens e.g. at 200x10 where CP-SAT hits its time limit before
    reaching the true optimum).

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

    # Common instances present in all seeds AND in c_cpsat
    common = set(per_seed_dicts[0].keys())
    for d in per_seed_dicts[1:]:
        common &= set(d.keys())
    if c_cpsat:
        common &= set(c_cpsat.keys())
    instances = sorted(common)

    if not instances:
        return np.array([]), []

    matrix = np.zeros((len(SEEDS), len(instances)))
    for i, s in enumerate(SEEDS):
        for j, inst in enumerate(instances):
            matrix[i, j] = c_cpsat[inst] / per_seed_dicts[i][inst]
    return matrix, instances


def build_baseline_score(baseline_makespans: dict[str, float], c_cpsat: dict[str, float],
                         instances: list[str]) -> np.ndarray:
    """Score array for a deterministic baseline (shape (1, num_instances))."""
    scores = np.array([c_cpsat[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
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


def analyze_scaling(score_dict_per_size: dict[str, dict[str, np.ndarray]],
                    baseline_scores_per_size: dict[str, dict[str, np.ndarray]],
                    sizes: list[str]) -> dict:
    """Bootstraps IQM vs. instance size, one series per (method, mode); plus baseline points."""
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])

    combos = [combo_key(method, mode) for method in METHODS for mode in MODES]
    methods_result = {}
    total_bs = len(combos) * len(sizes)
    bs_i = 0
    for key in combos:
        means, lows, highs = [], [], []
        for size in sizes:
            sd = score_dict_per_size.get(size, {})
            bs_i += 1
            if key not in sd:
                means.append(np.nan)
                lows.append(np.nan)
                highs.append(np.nan)
                continue
            print(f"    bootstrap {key} {size} ({bs_i}/{total_bs}) ...", end=" ", flush=True)
            t0 = time.time()
            single = {key: sd[key]}
            iqm_scores, iqm_cis = rly.get_interval_estimates(single, iqm_fn, reps=BOOTSTRAP_REPS)
            print(f"{time.time()-t0:.1f}s")
            means.append(float(iqm_scores[key][0]))
            lows.append(float(iqm_cis[key][0, 0]))
            highs.append(float(iqm_cis[key][1, 0]))
        methods_result[key] = {"means": means, "lows": lows, "highs": highs}

    baselines_result = {}
    for b in BASELINES:
        means = []
        for size in sizes:
            arr = baseline_scores_per_size.get(size, {}).get(b)
            means.append(np.nan if arr is None or arr.size == 0 else float(metrics.aggregate_iqm(arr)))
        if not all(np.isnan(means)):
            baselines_result[b] = means

    return {"methods": methods_result, "baselines": baselines_result}


def analyze_efficiency(sizes: list[str]) -> dict:
    """Solve time and per-decision forward-pass time vs. instance size, per
    method and inference mode (Hurink excluded -- ran on different hardware).

    Mean across seeds, with a min/max band. Coarsening overhead itself
    (avg_coarse_ms) is negligible for both methods and isn't part of the
    headline story, so it's left out.

    Keyed by combo_key(method, mode) so greedy and sampling can be drawn as
    separate series; the per-decision forward pass should be mode-independent
    (same network, same graph), so having both in the plot is what makes that
    visible rather than assumed.
    """
    result = {"sizes": sizes, "methods": {}}
    for method in METHODS:
        for mode in MODES:
            solve_mean, solve_lo, solve_hi = [], [], []
            fwd_mean, fwd_lo, fwd_hi = [], [], []
            for size in sizes:
                solve_per_seed, fwd_per_seed = [], []
                for s in SEEDS:
                    st = load_drl_solve_times(method, size, s, mode)
                    ov = load_drl_overhead(method, size, s, mode)
                    if st is not None and st.size:
                        solve_per_seed.append(st.mean())
                    if ov is not None and not ov.empty:
                        fwd_per_seed.append(ov["avg_forward_ms"].astype(float).mean())

                if solve_per_seed:
                    solve_mean.append(float(np.mean(solve_per_seed)))
                    solve_lo.append(float(np.min(solve_per_seed)))
                    solve_hi.append(float(np.max(solve_per_seed)))
                else:
                    solve_mean.append(np.nan)
                    solve_lo.append(np.nan)
                    solve_hi.append(np.nan)

                if fwd_per_seed:
                    fwd_mean.append(float(np.mean(fwd_per_seed)))
                    fwd_lo.append(float(np.min(fwd_per_seed)))
                    fwd_hi.append(float(np.max(fwd_per_seed)))
                else:
                    fwd_mean.append(np.nan)
                    fwd_lo.append(np.nan)
                    fwd_hi.append(np.nan)

            result["methods"][combo_key(method, mode)] = {
                "solve_time": {"mean": solve_mean, "lo": solve_lo, "hi": solve_hi},
                "forward_ms": {"mean": fwd_mean, "lo": fwd_lo, "hi": fwd_hi},
            }
    return result


def create_gap_table():
    """Table 6: Makespan and gap-to-CP-SAT per method x mode x instance size.

    Rows are instance sizes plus Mk (Brandimarte) and the Hurink datasets.
    Columns are CP-SAT, the four dispatching rules, and each (method, mode)
    combo. CP-SAT gets a Makespan and an "Optimal" sub-column ("n/N"
    instances solved to proven optimality, vs. hitting the time limit --
    blank for sizes/datasets where the CP-SAT CSV doesn't have a status
    column yet, currently Mk and the Hurink sets); the other columns get a
    Makespan and a Gap % (relative to CP-SAT) sub-column. Sizes/datasets
    without CP-SAT data get NaN gaps. Saved as .xlsx, not a plot, so it
    doesn't go through plot.py.
    """
    all_sizes = TEST_SIZES + [MK_SIZE] + HURINK_DATASETS
    combo_labels = [f"{METHOD_LABELS[m]} ({MODE_LABELS[mo]})" for m in METHODS for mo in MODES]
    col_methods = ["CPSAT"] + DISPATCHING_RULES + combo_labels
    col_tuples = []
    for m in col_methods:
        col_tuples.append((m, "Makespan"))
        if m == "CPSAT":
            col_tuples.append((m, "Optimal"))
        else:
            col_tuples.append((m, "Gap %"))
    columns = pd.MultiIndex.from_tuples(col_tuples)
    table = pd.DataFrame(index=pd.Index(all_sizes, name="size"), columns=columns)

    for size in all_sizes:
        baseline_data = get_baseline_makespans(size)
        cpsat_data = baseline_data.get("CPSAT")
        cpsat_status = load_cpsat_status(size)

        if cpsat_data:
            table.loc[size, ("CPSAT", "Makespan")] = np.mean(list(cpsat_data.values()))
            if cpsat_status:
                n_optimal = sum(1 for v in cpsat_status.values() if v == "OPTIMAL")
                table.loc[size, ("CPSAT", "Optimal")] = f"{n_optimal}/{len(cpsat_status)}"

        # Dispatching rules
        for rule in DISPATCHING_RULES:
            b_data = baseline_data.get(rule)
            if b_data is None:
                continue
            table.loc[size, (rule, "Makespan")] = np.mean(list(b_data.values()))
            if cpsat_data:
                common = sorted(set(b_data.keys()) & set(cpsat_data.keys()))
                if common:
                    gaps = [(b_data[i] / cpsat_data[i] - 1) * 100 for i in common]
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
                    if cpsat_data:
                        common = sorted(set(d.keys()) & set(cpsat_data.keys()))
                        if common:
                            gaps = [(d[i] / cpsat_data[i] - 1) * 100 for i in common]
                            per_seed_gaps.append(np.mean(gaps))
                if per_seed_makespans:
                    table.loc[size, (label, "Makespan")] = np.mean(per_seed_makespans)
                if per_seed_gaps:
                    table.loc[size, (label, "Gap %")] = np.mean(per_seed_gaps)

    xlsx_path = PLOTS_DIR / "06_gap_table.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="gap_table")
    print(f"  Saved 06_gap_table.xlsx ({len(table)} rows)")

    print()
    print(table.to_string())


# Main

def main():
    print("=" * 70)
    print("No Unpooling Analysis")
    print("=" * 70)
    print(f"Script dir:    {SCRIPT_DIR}")
    print(f"Benchmarks:    {BENCHMARKS_DIR}")
    print(f"Cache out:     {ANALYSIS_CACHE}")
    print(f"Methods:       {METHODS}")
    print(f"Modes:         {MODES}")
    print(f"Seeds:         {SEEDS}")
    print(f"Test sizes:    {TEST_SIZES}")
    print(f"Hurink:        {HURINK_DATASETS}")
    print(f"Baselines:     {BASELINES}")
    print()

    # 1. Prepare score matrices per size (TEST_SIZES + Hurink datasets share
    # the same loading/bootstrap machinery, just plotted separately later)
    print("Loading data and building score matrices ...")
    all_sizes = TEST_SIZES + HURINK_DATASETS
    score_dict_per_size = {}
    baseline_scores_per_size = {}

    for size in all_sizes:
        print(f"  Size {size}")
        baseline_data = get_baseline_makespans(size)
        c_cpsat = baseline_data.get("CPSAT")
        if not c_cpsat:
            print(f"  [warn] No CP-SAT data for {size}, skipping")
            continue

        score_dict = {}
        for method in METHODS:
            for mode in MODES:
                matrix, instances = build_score_matrix(method, size, c_cpsat, mode)
                if matrix.size == 0:
                    continue
                key = combo_key(method, mode)
                score_dict[key] = matrix
                print(f"    {key}: {matrix.shape}, IQM={np.mean(np.sort(matrix.flatten())[len(matrix.flatten())//4:3*len(matrix.flatten())//4]):.4f}")
        score_dict_per_size[size] = score_dict

        # Baselines as (1, num_instances) arrays
        if score_dict:
            # Use the instance list of the first available (method, mode) combo as reference
            first_key = next(iter(score_dict))
            first_method, first_mode = split_combo_key(first_key)
            _, instances = build_score_matrix(first_method, size, c_cpsat, first_mode)
            baseline_scores = {}
            for b, b_data in baseline_data.items():
                arr = build_baseline_score(b_data, c_cpsat, instances)
                if arr.size:
                    baseline_scores[b] = arr

            best_dr = compute_c_best_dr(baseline_data)
            arr = build_baseline_score(best_dr, c_cpsat, instances)
            if arr.size:
                baseline_scores["BestDR"] = arr

            baseline_scores_per_size[size] = baseline_scores

    cache = {}
    steps = [
        ("06 Gap Table",                                 create_gap_table, None),
        ("Training Curves (aggregate)",                  analyze_training_curves, "training_curves"),
        ("IQM Bars (bootstrap)",                          lambda: analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size, TEST_SIZES), "iqm_bars"),
        ("IQM Bars Hurink (bootstrap)",                   lambda: analyze_iqm_bars(score_dict_per_size, baseline_scores_per_size, HURINK_DATASETS), "iqm_bars_hurink"),
        ("Scaling (bootstrap)",                           lambda: analyze_scaling(score_dict_per_size, baseline_scores_per_size, TEST_SIZES), "scaling"),
        ("Efficiency (solve time / forward time)",        lambda: analyze_efficiency(EFFICIENCY_SIZES), "efficiency"),
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
