"""Shared configuration for the pilot test analysis and plotting scripts."""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent       # results/
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Cached bootstrap/analysis results, written by analyze.py and read by plot.py
ANALYSIS_CACHE = SCRIPT_DIR / "analysis_cache.pkl"

# Method names (folder name without "_20x10" and without ".pt")
METHODS = ["sagc", "nopooling"]
METHOD_LABELS = {"sagc": "SAGC", "nopooling": "NoPooling"}
# Cool hues, deliberately far from the warm DR family below so the two groups
# never get confused. Only 2 of these are used for now, leaving standard
# tab10-ish colors (green, orange, plain red, ...) free for future methods.
METHOD_COLORS = {"sagc": "#4C72B0", "nopooling": "#DD8452"}

# Seeds, 3 for the pilot
SEEDS = [0, 1, 2]

# Inference modes. Same method/color, distinguished by linestyle/marker in plots.
MODES = ["greedy", "sample"]
MODE_LABELS = {"greedy": "Greedy", "sample": "Sampling"}
MODE_LINESTYLES = {"greedy": "-", "sample": "--"}
MODE_MARKERS = {"greedy": "o", "sample": "^"}
MODE_HATCHES = {"greedy": "", "sample": "///"}  # for bar charts, where linestyle doesn't apply

# Test sizes, once as display label (with x) and once as folder name (without x).
# 9 sizes -> laid out as a 3x3 grid in the size-indexed plots.
TEST_SIZES = ["10x5", "15x10", "20x5", "20x10", "30x10", "40x10", "50x10", "100x10", "200x10"]
SIZE_FOLDER_MAP = {
    "10x5": "1005",
    "15x10": "1510",
    "20x5": "2005",
    "20x10": "2010",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
    "Mk": "brandimarte",
}

# Brandimarte separate, only for the gap table at the end
MK_SIZE = "Mk"

# Hurink datasets: not "sizes" in the scaling sense, run through the same
# bootstrap pipeline as TEST_SIZES but plotted separately (1x3 grid).
HURINK_DATASETS = ["edata", "rdata", "vdata"]
HURINK_LABELS = {"edata": "Edata", "rdata": "Rdata", "vdata": "Vdata"}
# Folder name equals the key, so SIZE_FOLDER_MAP just needs identity entries
# to keep analyze.py's SIZE_FOLDER_MAP[size] lookups working unchanged.
SIZE_FOLDER_MAP.update({d: d for d in HURINK_DATASETS})

# Sizes for the runtime/efficiency plot: sampling-only, so 200x10 is dropped
# (too slow to sample repeatedly, same reasoning as the 02/04 exclusions).
EFFICIENCY_SIZES = [s for s in TEST_SIZES if s != "200x10"]

# Baselines (dispatching rules). Add CP-SAT and GA later.
BASELINES = ["MWR", "SPT", "MOR", "FIFO"]
# One warm family (gold -> rust -> maroon), ordered by decreasing lightness so
# the group reads as related and stays visually distinct from METHOD_COLORS.
BASELINE_COLORS = {
    "MWR": "#555555",     # gold
    "SPT": "#A0785A",     # amber/orange
    "MOR": "#8A9A3B",     # rust
    "FIFO": "#5BA8A0",    # brick red
    "CPSAT": "#9A7090",   # later
    "GA": "#C4A832",      # later
}

# rliable bootstrap replications
BOOTSTRAP_REPS = 5000  # 50000 is standard, 5000 is enough for the pilot test


def combo_key(method: str, mode: str) -> str:
    """Composite key identifying a (method, mode) series, e.g. 'sagc__greedy'."""
    return f"{method}__{mode}"


def split_combo_key(key: str) -> tuple[str, str]:
    """Inverse of combo_key."""
    method, mode = key.split("__", 1)
    return method, mode
