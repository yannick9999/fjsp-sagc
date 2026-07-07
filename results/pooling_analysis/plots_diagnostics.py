"""
Pooling Diagnostics, Summary Plots and Score Table

Folder structure (relative to this script):
  ./data/seed{s}/{size_folder}_greedy/diagnostics_*.csv

Output:
  ./plots/table_pooling_scores.png
  ./plots/episode_critical_retention.png
  ./plots/episode_successor_retention.png
  ./plots/episode_slack_correlation.png
  ./plots/episode_frontier_dist.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data"
OUT_DIR    = SCRIPT_DIR / "plots"
SEEDS      = [0, 1, 2]

SIZE_FOLDER_MAP = {
    "10x5":  "1005",
    "20x5":  "2005",
    "15x10": "1510",
    "20x10": "2010",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10":"10010",
    "200x10":"20010",
}

SIZE_COLORS = {
    "10x5":  "#a8d8ea",
    "20x5":  "#7ec8e3",
    "15x10": "#2ecc71",
    "20x10": "#27ae60",
    "30x10": "#f39c12",
    "40x10": "#e67e22",
    "50x10": "#e74c3c",
    "100x10":"#c0392b",
    "200x10":"#8e44ad",
}

RANDOM_COLOR = "#95a5a6"
RANDOM_STYLE = dict(color=RANDOM_COLOR, linestyle="--", linewidth=1.8,
                    marker="s", markersize=5, label="Random (avg. all sizes)")
EPISODE_BINS = 10


# ── Data loading ──────────────────────────────────────────────────────────────

def load_size(size: str):
    folder = SIZE_FOLDER_MAP[size]
    frames = []
    for s in SEEDS:
        path = DATA_DIR / f"seed{s}" / f"{folder}_greedy"
        if not path.is_dir():
            continue
        files = sorted(path.glob("diagnostics_*.csv"))
        if not files:
            continue
        df = pd.read_csv(files[-1])
        df["seed"] = s
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all():
    data = {}
    for size in SIZE_FOLDER_MAP:
        df = load_size(size)
        if df is not None:
            data[size] = df
            print(f"  {size}: {len(df)} rows, "
                  f"{df['instance'].nunique()} instances, "
                  f"{df['seed'].nunique()} seeds")
        else:
            print(f"  {size}: no data found")
    return data


# ── Score computation ─────────────────────────────────────────────────────────

def instance_scores_delta(df, metric, normalize_by=None):
    """
    Per-instance: mean_steps(learned) - mean_steps(random).
    Optionally normalized by mean_steps(normalize_by) over all rows.
    Returns np.ndarray of per-instance scores.
    """
    scores = []
    for inst, grp in df.groupby("instance"):
        learned = grp[grp["method"] == "learned"][metric].dropna()
        random  = grp[grp["method"] == "random"][metric].dropna()
        if learned.empty or random.empty:
            continue
        diff = learned.mean() - random.mean()
        if normalize_by is not None:
            norm_val = grp[normalize_by].dropna().mean()
            if norm_val > 1e-8:
                diff /= norm_val
            else:
                continue
        scores.append(diff)
    return np.array(scores)


def instance_scores_learned(df, metric):
    """Per-instance mean of learned only. Used for slack_correlation."""
    scores = []
    for inst, grp in df.groupby("instance"):
        learned = grp[grp["method"] == "learned"][metric].dropna()
        if learned.empty:
            continue
        scores.append(learned.mean())
    return np.array(scores)


# ── Score table ───────────────────────────────────────────────────────────────

def compute_table(data):
    records = []
    for size, df in data.items():
        rec = {"size": size}

        s = instance_scores_delta(df, "critical_retention")
        rec["crit_mean"], rec["crit_std"] = (s.mean(), s.std()) if len(s) else (np.nan, np.nan)

        s = instance_scores_delta(df, "successor_retention")
        rec["succ_mean"], rec["succ_std"] = (s.mean(), s.std()) if len(s) else (np.nan, np.nan)

        s = instance_scores_delta(df, "mean_frontier_dist_kept_dr",
                                  normalize_by="mean_frontier_dist_all_dr")
        rec["front_mean"], rec["front_std"] = (s.mean(), s.std()) if len(s) else (np.nan, np.nan)

        s = instance_scores_learned(df, "slack_correlation")
        rec["slack_mean"], rec["slack_std"] = (s.mean(), s.std()) if len(s) else (np.nan, np.nan)

        records.append(rec)
    return pd.DataFrame(records).set_index("size")


def fmt(mean, std):
    if np.isnan(mean):
        return "n/a"
    return f"{mean:+.3f}\n\u00b1{std:.3f}"


def plot_table(table):
    sizes = list(table.index)

    row_labels = [
        "Critical retention\n(\u0394 learned\u2013random, higher is better)",
        "Successor retention\n(\u0394 learned\u2013random, higher is better)",
        "Frontier dist (norm.)\n(\u0394 learned\u2013random, lower is better)",
        "Slack correlation\n(learned only, lower is better)",
    ]
    prefixes = ["crit", "succ", "front", "slack"]

    cell_text = []
    for prefix in prefixes:
        row = []
        for size in sizes:
            if size not in table.index:
                row.append("n/a")
            else:
                row.append(fmt(table.loc[size, f"{prefix}_mean"],
                               table.loc[size, f"{prefix}_std"]))
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=sizes,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 3.0)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0 or c == -1:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#ecf0f1" if r % 2 == 0 else "white")

    ax.set_title("Pooling quality scores (mean +/- std over instances)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "table_pooling_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved table_pooling_scores.png")


# ── Episode helpers ───────────────────────────────────────────────────────────

def binned_mean(df, method, metric):
    sub = df[df["method"] == method].copy()
    sub["progress"] = sub.groupby("instance")["step"].transform(
        lambda s: s / max(s.max(), 1)
    )
    sub["bin"] = pd.cut(sub["progress"], bins=EPISODE_BINS,
                        labels=False, include_lowest=True)
    return sub.groupby("bin")[metric].mean()


def averaged_random(data, metric):
    """Average random curve across all sizes."""
    curves = [binned_mean(df, "random", metric) for df in data.values()]
    return pd.concat(curves, axis=1).mean(axis=1)


# ── Episode plots ─────────────────────────────────────────────────────────────

def plot_episode(data, metric, ylabel, title, fname, hline=None):
    fig, ax = plt.subplots(figsize=(8, 5))

    for size, df in data.items():
        learned = binned_mean(df, "learned", metric)
        ax.plot(learned.index * 10 + 5, learned.values,
                color=SIZE_COLORS[size], linewidth=2,
                marker="o", markersize=4, label=size)

    rnd = averaged_random(data, metric)
    ax.plot(rnd.index * 10 + 5, rnd.values, **RANDOM_STYLE)

    if hline is not None:
        ax.axhline(hline, color="black", linestyle=":", linewidth=1, zorder=0)

    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_episode_frontier(data):
    fig, ax = plt.subplots(figsize=(8, 5))

    for size, df in data.items():
        learned = binned_mean(df, "learned", "mean_frontier_dist_kept_dr")
        ax.plot(learned.index * 10 + 5, learned.values,
                color=SIZE_COLORS[size], linewidth=2,
                marker="o", markersize=4, label=size)

    rnd = averaged_random(data, "mean_frontier_dist_kept_dr")
    ax.plot(rnd.index * 10 + 5, rnd.values, **RANDOM_STYLE)

    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel("Mean frontier dist (kept, decision-relevant)")
    ax.set_title("Frontier distance of kept nodes over episode")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "episode_frontier_dist.png", dpi=150)
    plt.close(fig)
    print("  Saved episode_frontier_dist.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    data = load_all()
    if not data:
        print("No data found.")
        return

    print("\nComputing score table ...")
    table = compute_table(data)
    print(table.to_string())
    plot_table(table)

    print("\nGenerating episode plots ...")
    plot_episode(data, "critical_retention",
                 "Critical retention", "Critical retention over episode",
                 "episode_critical_retention.png")
    plot_episode(data, "successor_retention",
                 "Successor retention", "Successor retention over episode",
                 "episode_successor_retention.png")
    plot_episode(data, "slack_correlation",
                 "Slack correlation (Spearman)", "Slack correlation over episode",
                 "episode_slack_correlation.png", hline=0.0)
    plot_episode_frontier(data)

    print("\nDone.")


if __name__ == "__main__":
    main()