"""
Pooling-Diagnostics Plot-Skript

Datenstruktur (relativ zum Speicherort dieses Skripts):
  ./sagc_no_unpooling_20x10/test/seed{s}/{size_folder}_greedy/diagnostics_*.csv

Nur die "sagc"-Methode schreibt Pooling-Diagnostics (nopooling poolt nicht).
Fuer jede Groesse werden die CSVs aller verfuegbaren Seeds zusammengefuehrt
(pandas.concat, Spalte "seed" wird ergaenzt) und daraus ein Satz Plots erzeugt.
Manche Groessen (100x10, 200x10) haben nur fuer 5 statt 100 Instanzen
Diagnostics -- das Skript verlangt das nicht explizit, es aggregiert einfach
ueber alle Zeilen, die es in den gefundenen CSVs vorfindet, und meldet die
tatsaechliche Instanzanzahl beim Ausfuehren.

Ergebnisse (PNG-Plots, keine Tabelle) werden gespeichert unter:
  ./plots/diagnostics/{size}/dist_*.png
  ./plots/diagnostics/{size}/episode_*.png
  ./plots/diagnostics/{size}/kept_composition.png
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

# Testgroessen -> Ordnername (wie in analyze.py)
SIZE_FOLDER_MAP = {
    "10x5": "1005",
    "20x5": "2005",
    "15x10": "1510",
    "20x10": "2010",
    "30x10": "3010",
    "40x10": "4010",
    "50x10": "5010",
    "100x10": "10010",
    "200x10": "20010",
}


def find_diagnostics_csv(method_dir: Path, seed: int, size_folder: str) -> Path | None:
    """Findet die neueste diagnostics_*.csv fuer eine Groesse und einen Seed."""
    folder = method_dir / "test" / f"seed{seed}" / f"{size_folder}_greedy"
    if not folder.is_dir():
        return None
    files = sorted(folder.glob("diagnostics_*.csv"))
    return files[-1] if files else None


def load_combined_df(method_dir: Path, size_folder: str, seeds: list[int]) -> pd.DataFrame | None:
    """Laedt und kombiniert die Diagnostics-CSVs aller Seeds fuer eine Groesse.

    Jeder Seed liefert eigene Checkpoints/Trajektorien, es gibt keine
    ueberlappenden Zeilen, daher werden die Seeds einfach aneinandergehaengt
    (nicht gemittelt) um die Stichprobengroesse fuer die Plots zu erhoehen.
    Eine Spalte "seed" wird ergaenzt, falls man spaeter pro Seed aufschluesseln
    moechte.
    """
    frames = []
    for s in seeds:
        csv_path = find_diagnostics_csv(method_dir, s, size_folder)
        if csv_path is None:
            print(f"    [warn] keine diagnostics CSV fuer seed{s}")
            continue
        df = pd.read_csv(csv_path)
        df["seed"] = s
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


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
    parser = argparse.ArgumentParser(description="Plot pooling diagnostics from sagc_no_unpooling_20x10")
    parser.add_argument(
        "--method-dir", type=Path, default=SCRIPT_DIR / "sagc_no_unpooling_20x10",
        help="Ordner mit test/seed{s}/{size}_greedy/diagnostics_*.csv (default: sagc_no_unpooling_20x10)",
    )
    parser.add_argument(
        "--size", type=str, default=None, choices=list(SIZE_FOLDER_MAP.keys()),
        help="Nur eine Groesse auswerten, z.B. 20x10 (default: alle Groessen)",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--output", type=Path, default=SCRIPT_DIR / "plots" / "diagnostics",
        help="Basis-Output-Ordner, es wird pro Groesse ein Unterordner angelegt",
    )
    args = parser.parse_args()

    sizes = [args.size] if args.size else list(SIZE_FOLDER_MAP.keys())

    print(f"Method dir: {args.method_dir}")
    print(f"Seeds:      {args.seeds}")
    print(f"Output:     {args.output}")
    print()

    for size in sizes:
        size_folder = SIZE_FOLDER_MAP[size]
        print(f"[{size}] loading diagnostics ...")
        df = load_combined_df(args.method_dir, size_folder, args.seeds)
        if df is None:
            print(f"    [skip] keine diagnostics CSV fuer {size} gefunden")
            continue

        n_instances = df["instance"].nunique()
        n_seeds_found = df["seed"].nunique()
        print(f"    {len(df)} rows, {n_instances} instances, {n_seeds_found}/{len(args.seeds)} seeds")

        out_dir = args.output / size
        make_all_plots(df, out_dir)
        print(f"    Plots saved to {out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
