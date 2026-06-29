"""
Pooling diagnostics: compute per-step metrics for trained pooling models.
"""
import os
import random as pyrandom

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compute_node_categories(state, env, batch_idx=0):
    """
    Compute boolean masks for the four node categories of one instance.

    Returns dict with:
        is_padding   [N] bool
        is_completed [N] bool
        is_eligible  [N] bool
        is_critical  [N] bool   (open + slack == 0)
        is_open      [N] bool   (not padding, not completed)
        is_decision_relevant [N] bool   (open, not eligible, for slack/score corr)
        slack        [N] float  (raw, unnormalized)
        frontier_dist [N] float (op_index - current_step_of_job)
        opes_appertain [N] long
    """
    b = batch_idx
    N = state.feat_opes_batch.shape[-1]
    nums_opes = int(state.nums_opes_batch[b].item())
    device = state.feat_opes_batch.device

    is_padding = torch.arange(N, device=device) >= nums_opes

    ope_step = state.ope_step_batch[b]
    opes_appertain = state.opes_appertain_batch[b]
    current_step_per_op = ope_step[opes_appertain]
    op_indices = torch.arange(N, device=device)
    is_completed = (op_indices < current_step_per_op) & ~is_padding

    end_ope_biases = state.end_ope_biases_batch[b]
    ope_step_clamped = torch.where(ope_step > end_ope_biases, end_ope_biases, ope_step)
    num_jobs = ope_step.shape[0]
    num_mas = state.ope_ma_adj_batch.shape[-1]
    eligible_proc = state.ope_ma_adj_batch[b].gather(
        0, ope_step_clamped[:, None].expand(-1, num_mas))
    ma_eligible = ~state.mask_ma_procing_batch[b].unsqueeze(0).expand(num_jobs, num_mas)
    job_eligible = ~(state.mask_job_procing_batch[b] |
                     state.mask_job_finish_batch[b])[:, None].expand(-1, num_mas)
    eligible_pairs = job_eligible & ma_eligible & (eligible_proc == 1)
    job_truly_eligible = eligible_pairs.any(dim=-1)
    is_eligible = torch.zeros(N, dtype=torch.bool, device=device)
    is_eligible.scatter_(0, ope_step_clamped, job_truly_eligible)
    is_eligible = is_eligible & ~is_padding & ~is_completed

    slack = state.feat_opes_batch[b, 6, :].clone()
    critical_flag = state.feat_opes_batch[b, 7, :]
    is_open = ~is_padding & ~is_completed
    is_critical = (critical_flag > 0.5) & is_open

    is_decision_relevant = is_open & ~is_eligible

    frontier_dist = (op_indices.float() - current_step_per_op.float()).clamp(min=0)

    return {
        "N": N,
        "nums_opes": nums_opes,
        "is_padding": is_padding.cpu(),
        "is_completed": is_completed.cpu(),
        "is_eligible": is_eligible.cpu(),
        "is_critical": is_critical.cpu(),
        "is_open": is_open.cpu(),
        "is_decision_relevant": is_decision_relevant.cpu(),
        "slack": slack.cpu(),
        "frontier_dist": frontier_dist.cpu(),
        "opes_appertain": opes_appertain.cpu(),
    }


def compute_step_metrics(diag, cats, env, batch_idx=0):
    """
    Compute per-step pooling diagnostic metrics.

    Args:
        diag: dict from pool_layer._last_diag
        cats: dict from compute_node_categories
        env: FJSPEnv instance (for num_ope_biases_batch, nums_ope_batch)
        batch_idx: instance in the batch

    Returns:
        dict of scalar metrics for this step
    """
    b = batch_idx
    gate_scores = diag["gate_scores_raw"][b]
    top_idx = diag["top_idx"][b]
    k = diag["k"]
    N = cats["N"]
    nums_opes = cats["nums_opes"]

    is_padding = cats["is_padding"]
    is_completed = cats["is_completed"]
    is_eligible = cats["is_eligible"]
    is_critical = cats["is_critical"]
    is_open = cats["is_open"]
    is_decision_relevant = cats["is_decision_relevant"]
    slack = cats["slack"]
    frontier_dist = cats["frontier_dist"]

    kept_mask = torch.zeros(N, dtype=torch.bool)
    kept_mask[top_idx] = True

    # 1. Slack correlation
    if is_decision_relevant.sum() >= 3:
        scores_dr = gate_scores[is_decision_relevant].numpy()
        slack_dr = slack[is_decision_relevant].numpy()
        if np.std(scores_dr) > 1e-8 and np.std(slack_dr) > 1e-8:
            corr, _ = spearmanr(scores_dr, slack_dr)
            slack_corr = corr if not np.isnan(corr) else float("nan")
        else:
            slack_corr = float("nan")
    else:
        slack_corr = float("nan")

    # 2. Mean slack: kept vs discarded (decision-relevant)
    kept_dr = kept_mask & is_decision_relevant
    disc_dr = ~kept_mask & is_decision_relevant
    mean_slack_kept = slack[kept_dr].mean().item() if kept_dr.any() else float("nan")
    mean_slack_disc = slack[disc_dr].mean().item() if disc_dr.any() else float("nan")
    slack_diff_kept_minus_disc = mean_slack_kept - mean_slack_disc if (
        kept_dr.any() and disc_dr.any()) else float("nan")

    # 3. Critical path retention 
    n_critical_total = int(is_critical.sum().item())
    n_kept_critical = int((kept_mask & is_critical).sum().item())
    critical_retention = (n_kept_critical / n_critical_total) if n_critical_total > 0 else float("nan")

    # 4. Eligible retention (sanity check, should be 1.0) 
    n_eligible_total = int(is_eligible.sum().item())
    n_kept_eligible = int((kept_mask & is_eligible).sum().item())
    eligible_retention = (n_kept_eligible / n_eligible_total) if n_eligible_total > 0 else float("nan")

    # 5. Successor retention 
    num_ope_biases = env.num_ope_biases_batch[b].cpu()
    nums_ope = env.nums_ope_batch[b].cpu()
    num_jobs = num_ope_biases.shape[0]
    kept_set = set(top_idx.tolist())
    succ_total = 0
    succ_kept = 0
    for j in range(num_jobs):
        start = int(num_ope_biases[j].item())
        end = start + int(nums_ope[j].item())
        open_ops_in_job = [i for i in range(start, end) if is_open[i].item()]
        for idx in range(len(open_ops_in_job) - 1):
            op_curr = open_ops_in_job[idx]
            op_next = open_ops_in_job[idx + 1]
            if op_curr in kept_set:
                succ_total += 1
                if op_next in kept_set:
                    succ_kept += 1
    successor_retention = (succ_kept / succ_total) if succ_total > 0 else float("nan")

    # 6. Frontier distance: kept open nodes vs all open nodes 
    kept_open = kept_mask & is_open
    mean_frontier_dist_kept = frontier_dist[kept_open].mean().item() if kept_open.any() else float("nan")
    mean_frontier_dist_all = frontier_dist[is_open].mean().item() if is_open.any() else float("nan")
    kept_dr = kept_mask & is_decision_relevant
    mean_frontier_dist_kept_dr = frontier_dist[kept_dr].mean().item() if kept_dr.any() else float("nan")
    mean_frontier_dist_all_dr = frontier_dist[is_decision_relevant].mean().item() if is_decision_relevant.any() else float("nan")

    # 7. Kept composition (sanity check for masking) 
    n_kept = top_idx.shape[0]
    n_kept_completed = int((kept_mask & is_completed).sum().item())
    n_kept_padding = int((kept_mask & is_padding).sum().item())
    is_open_other = is_open & ~is_eligible & ~is_critical
    n_kept_open_other = int((kept_mask & is_open_other).sum().item())

    return {
        "k": k,
        "n_open": int(is_open.sum().item()),
        "n_eligible": n_eligible_total,
        "n_critical": n_critical_total,
        "n_decision_relevant": int(is_decision_relevant.sum().item()),
        "n_kept": n_kept,
        "n_kept_eligible": n_kept_eligible,
        "n_kept_critical": n_kept_critical,
        "n_kept_completed": n_kept_completed,
        "n_kept_padding": n_kept_padding,
        "n_kept_open_other": n_kept_open_other,
        # Main metrics
        "slack_correlation": slack_corr,
        "mean_slack_kept": mean_slack_kept,
        "mean_slack_discarded": mean_slack_disc,
        "slack_diff_kept_minus_disc": slack_diff_kept_minus_disc,
        "critical_retention": critical_retention,
        "eligible_retention": eligible_retention,
        "successor_retention": successor_retention,
        "mean_frontier_dist_kept": mean_frontier_dist_kept,
        "mean_frontier_dist_all": mean_frontier_dist_all,
        "mean_frontier_dist_kept_dr": mean_frontier_dist_kept_dr,
        "mean_frontier_dist_all_dr": mean_frontier_dist_all_dr,
    }


def save_graph_snapshot(diag, cats, state, env, batch_idx, step_idx,
                         instance_name, output_dir):
    """
    Save two side-by-side graph visualizations for one step:
    left: original graph before pooling, with gate scores as labels
    right: pooled graph (only kept nodes, with reconstructed chain edges)
    """
    from utils.visualize import plot_operation_graph

    b = batch_idx
    nums_opes = cats["nums_opes"]
    gate_scores = diag["gate_scores_raw"][b][:nums_opes]
    top_idx = diag["top_idx"][b]

    ope_pre_adj = state.ope_pre_adj_batch[b, :nums_opes, :nums_opes]
    opes_appertain = state.opes_appertain_batch[b, :nums_opes]
    is_eligible = cats["is_eligible"][:nums_opes]
    is_completed = cats["is_completed"][:nums_opes]
    is_critical = cats["is_critical"][:nums_opes]

    labels_gate = (gate_scores * 100).int()

    save_path_orig = os.path.join(output_dir,
        f"{instance_name}_step{step_idx:03d}_a_original.png")
    plot_operation_graph(
        adj=ope_pre_adj,
        opes_appertain=opes_appertain,
        nums_opes=nums_opes,
        labels=labels_gate,
        title=f"{instance_name} step {step_idx}: original (labels = gate score x 100)",
        save_path=save_path_orig,
        show=False,
        eligible_opes=is_eligible,
        completed_opes=is_completed,
    )

    k = top_idx.shape[0]
    kept_jobs = opes_appertain[top_idx]
    pooled_adj = torch.zeros(k, k)
    for i in range(k - 1):
        if kept_jobs[i] == kept_jobs[i + 1]:
            pooled_adj[i, i + 1] = 1.0
    pooled_labels = top_idx.int()
    pooled_eligible = cats["is_eligible"][top_idx]
    pooled_completed = cats["is_completed"][top_idx]

    save_path_pool = os.path.join(output_dir,
        f"{instance_name}_step{step_idx:03d}_b_pooled.png")
    plot_operation_graph(
        adj=pooled_adj,
        opes_appertain=kept_jobs,
        nums_opes=k,
        labels=pooled_labels,
        title=f"{instance_name} step {step_idx}: pooled (labels = original node id)",
        save_path=save_path_pool,
        show=False,
        eligible_opes=pooled_eligible,
        completed_opes=pooled_completed,
    )


# Plotting (run on the aggregated CSV after the test loop)

def plot_metric_distribution(df, metric, output_dir, title, ylim=None):
    """Boxplot of one metric across all steps."""
    vals = df[metric].dropna().values
    if len(vals) == 0:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(vals, widths=0.5, patch_artist=True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"dist_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_metric_over_episode(df, metric, output_dir, title, ylim=None):
    """Line plot of one metric over episode progress, binned into 10 buckets."""
    if "step" not in df.columns or len(df) == 0:
        return
    df = df.copy()
    df["progress"] = df.groupby("instance")["step"].transform(
        lambda s: s / max(s.max(), 1))
    df["bin"] = pd.cut(df["progress"], bins=10, labels=False, include_lowest=True)
    grouped = df.groupby("bin")[metric].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grouped.index * 10 + 5, grouped.values, marker="o", linewidth=2)
    ax.set_xlabel("Episode Progress (%)")
    ax.set_ylabel(metric)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"episode_{metric}.png"), dpi=150)
    plt.close(fig)


def plot_kept_composition(df, output_dir):
    """Stacked area chart of kept node composition over the episode."""
    if "step" not in df.columns or len(df) == 0:
        return
    df = df.copy()
    df["progress"] = df.groupby("instance")["step"].transform(
        lambda s: s / max(s.max(), 1))
    df["bin"] = pd.cut(df["progress"], bins=10, labels=False, include_lowest=True)
    cols = ["n_kept_eligible", "n_kept_critical", "n_kept_open_other",
            "n_kept_completed", "n_kept_padding"]
    grouped = df.groupby("bin")[cols].mean()
    labels = ["Eligible", "Critical Path", "Open (other)", "Completed", "Padding"]
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#95a5a6", "#bdc3c7"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(grouped.index * 10 + 5,
                  *[grouped[c].values for c in cols],
                  labels=labels, colors=colors, alpha=0.8)
    ax.set_xlabel("Episode Progress (%)")
    ax.set_ylabel("Number of Kept Nodes")
    ax.set_title("Kept Composition Over Episode")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "kept_composition.png"), dpi=150)
    plt.close(fig)


def make_all_plots(df, output_dir):
    """Generate all summary plots from the diagnostics dataframe."""
    os.makedirs(output_dir, exist_ok=True)

    plot_metric_distribution(df, "slack_correlation", output_dir,
                              "Slack Correlation (Spearman)", ylim=(-1.05, 1.05))
    plot_metric_distribution(df, "critical_retention", output_dir,
                              "Critical Path Retention", ylim=(-0.05, 1.05))
    plot_metric_distribution(df, "successor_retention", output_dir,
                              "Successor Retention", ylim=(-0.05, 1.05))
    plot_metric_distribution(df, "slack_diff_kept_minus_disc", output_dir,
                              "Slack Difference (Kept minus Discarded)")

    plot_metric_over_episode(df, "slack_correlation", output_dir,
                              "Slack Correlation Over Episode", ylim=(-1.05, 1.05))
    plot_metric_over_episode(df, "critical_retention", output_dir,
                              "Critical Path Retention Over Episode", ylim=(-0.05, 1.05))
    plot_metric_over_episode(df, "successor_retention", output_dir,
                              "Successor Retention Over Episode", ylim=(-0.05, 1.05))
    plot_metric_over_episode(df, "mean_frontier_dist_kept_dr", output_dir,
                              "Mean Frontier Distance, Kept (Decision-Relevant)")
    plot_metric_over_episode(df, "mean_frontier_dist_all_dr", output_dir,
                              "Mean Frontier Distance, All (Decision-Relevant)")

    plot_kept_composition(df, output_dir)
