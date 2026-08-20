"""Shared configuration for the 300x30 test analysis and plotting scripts."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent       # results/
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Cached bootstrap/analysis results, written by analyze.py and read by plot.py
ANALYSIS_CACHE = SCRIPT_DIR / "analysis_cache.pkl"

# Method names (folder name without "_test")
METHODS = ["sagc"]
METHOD_LABELS = {"sagc": "SAGC"}
METHOD_COLORS = {"sagc": "#4C72B0"}

# Seeds, 3 available
SEEDS = [0, 1, 2]

# Only greedy inference was run for this test.
MODES = ["greedy"]
MODE_LABELS = {"greedy": "Greedy"}
MODE_LINESTYLES = {"greedy": "-"}
MODE_MARKERS = {"greedy": "o"}
MODE_HATCHES = {"greedy": ""}

# Single test size.
TEST_SIZES = ["300x30"]
SIZE_FOLDER_MAP = {"300x30": "30030"}

# No CP-SAT data at this size (too large to solve), so there's no ground
# truth to normalize against. The dispatching rules are used as the
# reference instead -- see compute_c_best_dr in common_analysis logic.
DISPATCHING_RULES = ["MWR", "SPT", "MOR", "FIFO"]
BASELINES = DISPATCHING_RULES

BASELINE_COLORS = {
    "MWR": "#555555",
    "SPT": "#A0785A",
    "MOR": "#8A9A3B",
    "FIFO": "#5BA8A0",
    "BestDR": "#2E7D6B",
}
BASELINE_LABELS = {
    "MWR": "MWR", "SPT": "SPT", "MOR": "MOR", "FIFO": "FIFO",
    "BestDR": "Best DR",
}

# rliable bootstrap replications
BOOTSTRAP_REPS = 5000


def combo_key(method: str, mode: str) -> str:
    """Composite key identifying a (method, mode) series, e.g. 'sagc__greedy'."""
    return f"{method}__{mode}"


def split_combo_key(key: str) -> tuple[str, str]:
    """Inverse of combo_key."""
    method, mode = key.split("__", 1)
    return method, mode
