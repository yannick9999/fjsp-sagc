"""Shared configuration for the multi-size training analysis and plotting
scripts.

Unlike pilot_test/no_unpooling, a single model here is trained on a mix of
sizes (20x10/30x10/40x10) and evaluated using one of two checkpoints: the one
that scored best on in-distribution (indist) validation instances, or the one
that scored best on out-of-distribution (ood) validation instances. Both
checkpoints are run against the exact same test instances for every size --
indist/ood is a *checkpoint choice*, not a test-data split.

indist and ood are treated as two separate experiments (own cache, own plots
subfolder), since that's how the underlying question reads: "how well does
each checkpoint-selection strategy generalize across sizes". The SAGC-only
instance-structure ablation (flexibility/job-correlation sweep at 30x10) is a
third, unrelated experiment with its own scripts (ablation_analyze.py /
ablation_plot.py).
"""

from __future__ import annotations

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent       # results/
BENCHMARKS_DIR = RESULTS_DIR / "benchmarks"
PLOTS_DIR = SCRIPT_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Method names, and the on-disk folder for each (folder holds seed{n}/ for
# training and seed{n}/test/... for testing).
METHODS = ["sagc", "nopooling"]
METHOD_LABELS = {"sagc": "SAGC", "nopooling": "NoPooling"}
METHOD_DIRS = {"sagc": "multi_size_training_sagc", "nopooling": "multi_size_training_nopooling"}
# Cool hues, deliberately far from the warm DR family below so the two groups
# never get confused.
METHOD_COLORS = {"sagc": "#4C72B0", "nopooling": "#DD8452"}

# Seeds, 3 for the pilot
SEEDS = [0, 1, 2]

# Inference modes. Same method/color, distinguished by linestyle/marker in plots.
MODES = ["greedy", "sample"]
MODE_LABELS = {"greedy": "Greedy", "sample": "Sampling"}
MODE_LINESTYLES = {"greedy": "-", "sample": "--"}
MODE_MARKERS = {"greedy": "o", "sample": "^"}
MODE_HATCHES = {"greedy": "", "sample": "///"}  # for bar charts, where linestyle doesn't apply

# Checkpoint-selection splits -- which "save_best_*.pt" was used at test time.
# Run as two separate experiments (own cache, own plots subfolder), not as
# another combo axis like MODES.
SPLITS = ["indist", "ood"]
SPLIT_LABELS = {"indist": "In-Distribution", "ood": "Out-of-Distribution"}


def cache_path(split: str) -> Path:
    return SCRIPT_DIR / f"analysis_cache_{split}.pkl"


def plots_dir(split: str) -> Path:
    d = PLOTS_DIR / split
    d.mkdir(parents=True, exist_ok=True)
    return d


# Test sizes actually generalize across *both* jobs and machines here (not
# just jobs like pilot_test/no_unpooling), once as display label and once as
# folder name.
TEST_SIZES = ["10x10", "20x5", "20x10", "20x20", "20x30", "30x10", "40x10", "50x10", "100x10", "200x10"]
SIZE_FOLDER_MAP = {
    "10x10": "1010",
    "20x5": "2005",
    "20x10": "2010",
    "20x20": "2020",
    "20x30": "2030",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
}
# No Brandimarte (Mk) here -- it wasn't part of the multi-size test sweep.

# TEST_SIZES split by which dimension is swept, for the IQM bar chart --
# combining both sweeps in one plot made machines-fixed and jobs-fixed
# comparisons hard to read side by side.
JOB_SWEEP_SIZES = ["10x10", "20x10", "30x10", "40x10", "50x10", "100x10", "200x10"]     # machines=10, jobs vary
MACHINE_SWEEP_SIZES = ["20x5", "20x10", "20x20", "20x30"]                               # jobs=20, machines vary

# Hurink datasets: not "sizes" in the scaling sense, run through the same
# bootstrap pipeline as TEST_SIZES but plotted separately.
HURINK_DATASETS = ["edata", "rdata", "vdata"]
HURINK_LABELS = {"edata": "Edata", "rdata": "Rdata", "vdata": "Vdata"}
# Folder name equals the key, so SIZE_FOLDER_MAP just needs identity entries
# to keep analyze.py's SIZE_FOLDER_MAP[size] lookups working unchanged.
SIZE_FOLDER_MAP.update({d: d for d in HURINK_DATASETS})

# Sizes for the runtime/efficiency plot: sampling-only, so 200x10 is dropped
# (no sampling runs exist for 200x10 at all -- too slow).
EFFICIENCY_SIZES = [s for s in TEST_SIZES if s != "200x10"]

# Baselines (dispatching rules + CP-SAT). Add GA later.
BASELINES = ["MWR", "SPT", "MOR", "FIFO", "CPSAT"]
# The pure dispatching rules, without CP-SAT -- used for the "best dispatching
# rule" pseudo-baseline in the IQM bar charts.
DISPATCHING_RULES = ["MWR", "SPT", "MOR", "FIFO"]
# One warm family (gold -> rust -> maroon), ordered by decreasing lightness so
# the group reads as related and stays visually distinct from METHOD_COLORS.
BASELINE_COLORS = {
    "MWR": "#555555",     # gold
    "SPT": "#A0785A",     # amber/orange
    "MOR": "#8A9A3B",     # rust
    "FIFO": "#5BA8A0",    # brick red
    "CPSAT": "#9A7090",
    "BestDR": "#2E7D6B",  # best of the four dispatching rules, per instance -- teal, deliberately outside both the warm DR family and the cool method family so it reads as its own category
    "GA": "#C4A832",      # later
}
BASELINE_LABELS = {
    "MWR": "MWR", "SPT": "SPT", "MOR": "MOR", "FIFO": "FIFO",
    "CPSAT": "CP-SAT", "BestDR": "Best DR", "GA": "GA",
}

# rliable bootstrap replications
BOOTSTRAP_REPS = 5000  # 50000 is standard, 5000+ for real numbers, 100 while iterating


def combo_key(method: str, mode: str) -> str:
    """Composite key identifying a (method, mode) series, e.g. 'sagc__greedy'."""
    return f"{method}__{mode}"


def split_combo_key(key: str) -> tuple[str, str]:
    """Inverse of combo_key."""
    method, mode = key.split("__", 1)
    return method, mode
