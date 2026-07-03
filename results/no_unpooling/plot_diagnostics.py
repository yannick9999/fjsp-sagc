import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_distribution(df, metric, output_dir, title, ylim=None):
    df_learned = df[df["method"] == "learned"][metric].dropna()
    df_random = df[df["method"] == "random"][metric].dropna()

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [df_learned.values, df_random.values],
        labels=["Learned", "Random"],
        widths=0.5,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor("#2980b9")
    bp["boxes"][1].set_facecolor("#95a5a6")

    ax.set_title(title)
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dist_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_metric_over_episode(df, metric, output_dir, title, ylim=None):
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, color, marker in [("learned", "#2980b9", "o"), ("random", "#95a5a6", "s")]:
        sub = df[df["method"] == method].copy()
        sub["progress"] = sub.groupby("instance")["step"].transform(
            lambda s: s / max(s.max(), 1)
        )
        sub["bin"] = pd.cut(sub["progress"], bins=10, labels=False, include_lowest=True)
        grouped = sub.groupby("bin")[metric].mean()
        ax.plot(
            grouped.index * 10 + 5,
            grouped.values,
            marker=marker,
            linewidth=2,
            color=color,
            label=method.capitalize(),
        )

    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"episode_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_kept_composition(df, output_dir):
    df_learned = df[df["method"] == "learned"].copy()
    df_learned["progress"] = df_learned.groupby("instance")["step"].transform(
        lambda s: s / max(s.max(), 1)
    )
    df_learned["bin"] = pd.cut(
        df_learned["progress"], bins=10, labels=False, include_lowest=True
    )

    cols = [
        "n_kept_eligible",
        "n_kept_critical",
        "n_kept_open_other",
        "n_kept_completed",
        "n_kept_padding",
    ]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6", "#bdc3c7"]
    labels = ["Eligible", "Critical", "Open other", "Completed", "Padding"]

    grouped = df_learned.groupby("bin")[cols].mean()

    x = grouped.index * 10 + 5
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(x, [grouped[c].values for c in cols], labels=labels, colors=colors)
    ax.legend(loc="upper left")
    ax.set_title("Kept node composition over episode (learned)")
    ax.set_xlabel("Episode progress (%)")
    ax.set_ylabel("Mean node count")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "kept_composition.png"), dpi=150)
    plt.close(fig)


def make_all_plots(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    dist_specs = [
        ("slack_correlation", "Slack correlation distribution", (-1.05, 1.05)),
        ("critical_retention", "Critical retention distribution", (-0.05, 1.05)),
        ("successor_retention", "Successor retention distribution", (-0.05, 1.05)),
        ("slack_diff_kept_minus_disc", "Slack diff (kept - discarded) distribution", None),
    ]
    for metric, title, ylim in dist_specs:
        if metric in df.columns:
            plot_metric_distribution(df, metric, output_dir, title, ylim=ylim)

    episode_specs = [
        ("slack_correlation", "Slack correlation over episode"),
        ("critical_retention", "Critical retention over episode"),
        ("successor_retention", "Successor retention over episode"),
        ("mean_frontier_dist_kept_dr", "Mean frontier dist (kept) over episode"),
        ("mean_frontier_dist_all_dr", "Mean frontier dist (all) over episode"),
    ]
    for metric, title in episode_specs:
        if metric in df.columns:
            plot_metric_over_episode(df, metric, output_dir, title)

    composition_cols = [
        "n_kept_eligible", "n_kept_critical", "n_kept_open_other",
        "n_kept_completed", "n_kept_padding",
    ]
    if all(c in df.columns for c in composition_cols):
        plot_kept_composition(df, output_dir)


def main():
    parser = argparse.ArgumentParser(description="Plot pooling diagnostics from CSV")
    parser.add_argument("--csv", type=str, required=True, help="Path to diagnostics CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory for plots")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    make_all_plots(df, args.output)
    print(f"Plots saved to {args.output}")


if __name__ == "__main__":
    main()
