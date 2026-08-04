"""Instance-structure ablation: how flexibility (avg. eligible machines per
operation) and job-correlation (correlation between a job's own operation
times) affect SAGC, at a fixed size (30x10).

SAGC-only -- NoPooling wasn't run on this sweep. Uses the "indist" checkpoint
only (no "ood" variant exists for these folders). Each combo has its own
50-instance test set (not the same instances as the main 30x10 test set), and
there's no CP-SAT baseline for them, only the four dispatching rules -- so
C_best here is the best-dispatching-rule value, not CP-SAT.

This is an independent experiment from analyze.py (different instances,
different question), so it gets its own cache and plots/instance_structure/
subfolder, following the same indist/ood pattern.

Run this whenever the underlying data changes. Run instance_structure_plot.py (no
recomputation) whenever only the plot styling should change.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from rliable import library as rly
from rliable import metrics

from analyze import compute_c_best
from common import (
    BENCHMARKS_DIR,
    BOOTSTRAP_REPS,
    DISPATCHING_RULES,
    METHOD_DIRS,
    MODE_LABELS,
    MODES,
    SCRIPT_DIR,
    SEEDS,
    cache_path,
    plots_dir,
)

SPLIT_NAME = "instance_structure"
SIZE_TOKEN = "3010"  # 30x10, the only size this ablation was run at

FLEX_VALUES = ["115", "200", "500"]
FLEX_LABELS = {"115": "1.15", "200": "2.00", "500": "5.00"}
CORR_VALUES = ["000", "066"]
CORR_LABELS = {"000": "0.00", "066": "0.66"}
COMBOS = [(f, c) for f in FLEX_VALUES for c in CORR_VALUES]


def combo_folder(f: str, c: str) -> str:
    return f"{SIZE_TOKEN}_f{f}_c{c}"


def combo_key(f: str, c: str, mode: str) -> str:
    return f"f{f}_c{c}__{mode}"


def split_combo_key(key: str) -> tuple[str, str, str]:
    fc, mode = key.split("__", 1)
    f, c = fc[1:].split("_c", 1)
    return f, c, mode


# Data loading

def _find_excel(folder: Path, pattern: str = "*.xlsx"):
    if not folder.is_dir():
        return None
    files = sorted(folder.glob(pattern))
    return files[-1] if files else None


def load_sagc_makespans(f: str, c: str, seed: int, mode: str) -> dict[str, float] | None:
    """Loads test makespans for one (flexibility, correlation, mode, seed)."""
    folder = SCRIPT_DIR / METHOD_DIRS["sagc"] / "test" / f"seed{seed}" / f"{combo_folder(f, c)}_{mode}_indist"
    excel = _find_excel(folder, "test_results_*.xlsx")
    if excel is None:
        return None
    df = pd.read_excel(excel, sheet_name="makespan")
    instance_col = df.columns[0]
    makespan_col = df.columns[1]
    return dict(zip(df[instance_col].astype(str), df[makespan_col].astype(float)))


def load_baseline_makespans(rule: str, f: str, c: str) -> dict[str, float] | None:
    csv = BENCHMARKS_DIR / rule / f"{combo_folder(f, c)}.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    return dict(zip(df["instance_name"].astype(str), df["makespan"].astype(float)))


def get_baseline_makespans(f: str, c: str) -> dict[str, dict[str, float]]:
    result = {}
    for rule in DISPATCHING_RULES:
        m = load_baseline_makespans(rule, f, c)
        if m is not None:
            result[rule] = m
        else:
            print(f"  [warn] Baseline {rule} missing for {combo_folder(f, c)}")
    return result


def build_score_matrix(f: str, c: str, mode: str, c_best: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    """Score = C_best (best dispatching rule) / C_sagc (higher = better)."""
    per_seed_dicts = []
    for s in SEEDS:
        d = load_sagc_makespans(f, c, s, mode)
        if d is None:
            print(f"  [warn] Test data missing: {combo_folder(f, c)} {mode} seed{s}")
            return np.array([]), []
        per_seed_dicts.append(d)

    common = set(per_seed_dicts[0].keys())
    for d in per_seed_dicts[1:]:
        common &= set(d.keys())
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
    scores = np.array([c_best[i] / baseline_makespans[i] for i in instances if i in baseline_makespans])
    return scores.reshape(1, -1)


def analyze_iqm_bars() -> dict:
    """Bootstraps IQM + 95% CI for every (flexibility, correlation, mode)
    combo, all in a single rliable call so the CIs are computed jointly.
    """
    score_dict = {}
    baseline_scores = {}  # per combo (f, c): {rule: (1, n)} for reference

    for f, c in COMBOS:
        baseline_data = get_baseline_makespans(f, c)
        c_best = compute_c_best(baseline_data)
        if not c_best:
            print(f"  [warn] No baseline data for {combo_folder(f, c)}, skipping")
            continue

        instances_ref = None
        for mode in MODES:
            matrix, instances = build_score_matrix(f, c, mode, c_best)
            if matrix.size == 0:
                continue
            score_dict[combo_key(f, c, mode)] = matrix
            instances_ref = instances

        if instances_ref:
            per_rule = {}
            for rule, b_data in baseline_data.items():
                arr = build_baseline_score(b_data, c_best, instances_ref)
                if arr.size:
                    per_rule[rule] = float(metrics.aggregate_iqm(arr))
            baseline_scores[(f, c)] = per_rule

    print(f"  bootstrapping {len(score_dict)} combos jointly ...", end=" ", flush=True)
    t0 = time.time()
    iqm_fn = lambda x: np.array([metrics.aggregate_iqm(x)])
    iqm_scores, iqm_cis = rly.get_interval_estimates(score_dict, iqm_fn, reps=BOOTSTRAP_REPS)
    print(f"{time.time()-t0:.1f}s")

    means = {k: float(iqm_scores[k][0]) for k in score_dict}
    cis = {k: (float(iqm_cis[k][0, 0]), float(iqm_cis[k][1, 0])) for k in score_dict}

    return {"combos": COMBOS, "means": means, "cis": cis, "baseline_iqm": baseline_scores}


def create_gap_table():
    """Makespan and gap-to-best-dispatching-rule per (flexibility,
    correlation) combo. No CP-SAT here (not run on these instances), so the
    anchor column is 'Best DR' (min over the four dispatching rules) instead.
    """
    row_labels = [f"f={FLEX_LABELS[f]}, c={CORR_LABELS[c]}" for f, c in COMBOS]
    combo_labels = [f"SAGC ({MODE_LABELS[m]})" for m in MODES]
    col_methods = ["Best DR"] + DISPATCHING_RULES + combo_labels
    columns = pd.MultiIndex.from_product([col_methods, ["Makespan", "Gap %"]])
    table = pd.DataFrame(index=pd.Index(row_labels, name="combo"), columns=columns, dtype=float)

    for (f, c), row_label in zip(COMBOS, row_labels):
        baseline_data = get_baseline_makespans(f, c)
        c_best = compute_c_best(baseline_data)
        if not c_best:
            continue

        table.loc[row_label, ("Best DR", "Makespan")] = np.mean(list(c_best.values()))
        table.loc[row_label, ("Best DR", "Gap %")] = 0.0

        for rule in DISPATCHING_RULES:
            b_data = baseline_data.get(rule)
            if b_data is None:
                continue
            table.loc[row_label, (rule, "Makespan")] = np.mean(list(b_data.values()))
            common_inst = sorted(set(b_data.keys()) & set(c_best.keys()))
            if common_inst:
                gaps = [(b_data[i] / c_best[i] - 1) * 100 for i in common_inst]
                table.loc[row_label, (rule, "Gap %")] = np.mean(gaps)

        for mode in MODES:
            label = f"SAGC ({MODE_LABELS[mode]})"
            per_seed_makespans, per_seed_gaps = [], []
            for s in SEEDS:
                d = load_sagc_makespans(f, c, s, mode)
                if d is None:
                    continue
                common_inst = sorted(set(d.keys()) & set(c_best.keys()))
                if not common_inst:
                    continue
                per_seed_makespans.append(np.mean([d[i] for i in common_inst]))
                gaps = [(d[i] / c_best[i] - 1) * 100 for i in common_inst]
                per_seed_gaps.append(np.mean(gaps))
            if per_seed_makespans:
                table.loc[row_label, (label, "Makespan")] = np.mean(per_seed_makespans)
            if per_seed_gaps:
                table.loc[row_label, (label, "Gap %")] = np.mean(per_seed_gaps)

    out_dir = plots_dir(SPLIT_NAME)
    xlsx_path = out_dir / "06_gap_table.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="gap_table")
    print(f"  Saved {SPLIT_NAME}/06_gap_table.xlsx ({len(table)} rows)")

    print()
    print(table.to_string())


def main():
    print("=" * 70)
    print("Instance-Structure Ablation Analysis (SAGC, 30x10)")
    print("=" * 70)
    print(f"Script dir: {SCRIPT_DIR}")
    print(f"Combos:     {[combo_folder(f, c) for f, c in COMBOS]}")
    print(f"Modes:      {MODES}")
    print(f"Seeds:      {SEEDS}")
    print()

    import pickle

    cache = {}
    steps = [
        ("Gap Table",     create_gap_table, None),
        ("IQM Bars",      analyze_iqm_bars, "iqm_bars"),
    ]
    for i, (name, fn, key) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name} ...")
        out = fn()
        if key is not None:
            cache[key] = out

    path = cache_path(SPLIT_NAME)
    with open(path, "wb") as fobj:
        pickle.dump(cache, fobj)
    print(f"\nSaved analysis cache to {path}")
    print("All done. Run instance_structure_plot.py to (re)generate plots.")


if __name__ == "__main__":
    main()
