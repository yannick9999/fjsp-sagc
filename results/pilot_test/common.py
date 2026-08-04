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

# Test sizes, once as display label (with x) and once as folder name (without x)
TEST_SIZES = ["10x5", "15x10", "20x5", "20x10", "50x10", "100x10", "200x10"]
SIZE_FOLDER_MAP = {
    "10x5": "1005",
    "15x10": "1510",
    "20x5": "2005",
    "20x10": "2010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
    "Mk": "brandimarte",
}

# Brandimarte separate, only for the gap table at the end
MK_SIZE = "Mk"

# Baselines (dispatching rules + CP-SAT). Add GA later.
BASELINES = ["MWR", "SPT", "MOR", "FIFO", "CPSAT"]
# The pure dispatching rules, without CP-SAT -- used for the "best dispatching
# rule" pseudo-baseline in the IQM bar chart.
DISPATCHING_RULES = ["MWR", "SPT", "MOR", "FIFO"]
# One warm family (gold -> rust -> maroon), ordered by decreasing lightness so
# the group reads as related and stays visually distinct from METHOD_COLORS.
BASELINE_COLORS = {
    "MWR": "#555555",     # gold
    "SPT": "#A0785A",     # amber/orange
    "MOR": "#8A9A3B",     # rust
    "FIFO": "#5BA8A0",    # brick red
    "CPSAT": "#9A7090",
    "BestDR": "#2E2E2E",  # best of the four dispatching rules, per instance
    "GA": "#C4A832",      # later
}
BASELINE_LABELS = {
    "MWR": "MWR", "SPT": "SPT", "MOR": "MOR", "FIFO": "FIFO",
    "CPSAT": "CP-SAT", "BestDR": "Best DR", "GA": "GA",
}

# rliable bootstrap replications
BOOTSTRAP_REPS = 100  # 50000 is standard, 5000 is enough for the pilot test
