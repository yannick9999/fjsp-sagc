import numpy as np
import torch
from scipy.stats import spearmanr


def compute_node_categories(state, env, batch_idx=0):
    """
    Classify every node in one FJSP instance into diagnostic categories.

    Returns a dict with:
        N, nums_opes
        is_padding           [N] bool
        is_completed         [N] bool
        is_eligible          [N] bool
        is_critical          [N] bool   (open + critical-path flag)
        is_open              [N] bool   (not padding, not completed)
        is_decision_relevant [N] bool   (open, not eligible)
        slack                [N] float  (raw, unnormalized)
        frontier_dist        [N] float  (op_index - current_step_of_job, clamped >= 0)
        opes_appertain       [N] long
    All tensors are on CPU.
    """
    _ = env  # unused; kept so all three functions share the same (state, env, batch_idx) signature
    b = batch_idx
    N = state.feat_opes_batch.shape[-1]
    nums_opes = int(state.nums_opes_batch[b].item())
    device = state.feat_opes_batch.device

    op_indices = torch.arange(N, device=device)
    is_padding = op_indices >= nums_opes

    opes_appertain = state.opes_appertain_batch[b]          # [N]
    ope_step = state.ope_step_batch[b]                      # [J]
    end_ope_biases = state.end_ope_biases_batch[b]          # [J]

    # Clamp so finished jobs don't produce out-of-bounds indices
    ope_step_clamped = torch.where(ope_step > end_ope_biases, end_ope_biases, ope_step)

    # Completion: op index is strictly behind its job's current frontier
    current_step_per_op = ope_step[opes_appertain]          # [N] (unclamped for comparison)
    is_completed = (op_indices < current_step_per_op) & ~is_padding
    is_open = ~is_padding & ~is_completed

    # Eligible: frontier op of a job that has at least one valid (job, machine) pair
    num_jobs = ope_step.shape[0]
    num_mas = state.ope_ma_adj_batch.shape[-1]
    eligible_proc = state.ope_ma_adj_batch[b].gather(
        0, ope_step_clamped[:, None].expand(-1, num_mas))   # [J, M]
    ma_eligible = ~state.mask_ma_procing_batch[b].unsqueeze(0).expand(num_jobs, num_mas)
    job_eligible = ~(state.mask_job_procing_batch[b] |
                     state.mask_job_finish_batch[b])[:, None].expand(-1, num_mas)
    eligible_pairs = job_eligible & ma_eligible & (eligible_proc == 1)
    job_truly_eligible = eligible_pairs.any(dim=-1)         # [J]

    is_eligible = torch.zeros(N, dtype=torch.bool, device=device)
    is_eligible.scatter_(0, ope_step_clamped, job_truly_eligible)
    is_eligible = is_eligible & is_open                     # never eligible if completed/padding

    slack = state.feat_opes_batch[b, 6, :].clone()         # feature index 6
    is_critical = (state.feat_opes_batch[b, 7, :] > 0.5) & is_open  # feature index 7
    is_decision_relevant = is_open & ~is_eligible

    frontier_dist = (op_indices.float() - current_step_per_op.float()).clamp(min=0.0)

    return {
        "N": N,
        "nums_opes": nums_opes,
        "is_padding":           is_padding.cpu(),
        "is_completed":         is_completed.cpu(),
        "is_eligible":          is_eligible.cpu(),
        "is_critical":          is_critical.cpu(),
        "is_open":              is_open.cpu(),
        "is_decision_relevant": is_decision_relevant.cpu(),
        "slack":                slack.cpu(),
        "frontier_dist":        frontier_dist.cpu(),
        "opes_appertain":       opes_appertain.cpu(),
    }


def compute_step_metrics(diag, cats, env, batch_idx=0):
    """
    Compute per-step pooling quality metrics.

    Args:
        diag:      dict from pool_layer._last_diag
                   keys: gate_scores_raw [B, N], top_idx [B, k], k (int)
        cats:      dict from compute_node_categories (all tensors on CPU)
        env:       FJSPEnv instance (for num_ope_biases_batch, nums_ope_batch)
        batch_idx: which instance in the batch to evaluate

    Returns a flat dict of scalar metrics.
    """
    b = batch_idx
    gate_scores = diag["gate_scores_raw"][b]   # [N], CPU
    top_idx = diag["top_idx"][b]               # [k], CPU
    k = diag["k"]

    N = cats["N"]
    is_padding           = cats["is_padding"]
    is_completed         = cats["is_completed"]
    is_eligible          = cats["is_eligible"]
    is_critical          = cats["is_critical"]
    is_open              = cats["is_open"]
    is_decision_relevant = cats["is_decision_relevant"]
    slack                = cats["slack"]
    frontier_dist        = cats["frontier_dist"]

    kept_mask = torch.zeros(N, dtype=torch.bool)
    kept_mask[top_idx] = True

    # --- 1. Spearman correlation: gate scores vs slack (decision-relevant only) ---
    slack_corr = float("nan")
    if is_decision_relevant.sum() >= 3:
        scores_dr = gate_scores[is_decision_relevant].numpy()
        slack_dr  = slack[is_decision_relevant].numpy()
        if np.std(scores_dr) > 1e-8 and np.std(slack_dr) > 1e-8:
            corr, _ = spearmanr(scores_dr, slack_dr)
            slack_corr = float(corr) if not np.isnan(corr) else float("nan")

    # --- 2. Mean slack: kept vs discarded (decision-relevant) ---
    kept_dr = kept_mask & is_decision_relevant
    disc_dr = ~kept_mask & is_decision_relevant
    mean_slack_kept = slack[kept_dr].mean().item() if kept_dr.any() else float("nan")
    mean_slack_disc = slack[disc_dr].mean().item() if disc_dr.any() else float("nan")
    slack_diff_kept_minus_disc = (mean_slack_kept - mean_slack_disc
                                  if (kept_dr.any() and disc_dr.any()) else float("nan"))

    # --- 3. Critical-path retention ---
    n_critical = int(is_critical.sum().item())
    n_kept_critical = int((kept_mask & is_critical).sum().item())
    critical_retention = (n_kept_critical / n_critical) if n_critical > 0 else float("nan")

    # --- 4. Eligible retention (sanity check: should always be 1.0) ---
    n_eligible = int(is_eligible.sum().item())
    n_kept_eligible = int((kept_mask & is_eligible).sum().item())
    eligible_retention = (n_kept_eligible / n_eligible) if n_eligible > 0 else float("nan")

    # --- 5. Successor retention ---
    num_ope_biases = env.num_ope_biases_batch[b].cpu()
    nums_ope       = env.nums_ope_batch[b].cpu()
    num_jobs = num_ope_biases.shape[0]
    kept_set = set(top_idx.tolist())
    succ_total = 0
    succ_kept  = 0
    for j in range(num_jobs):
        start = int(num_ope_biases[j].item())
        end   = start + int(nums_ope[j].item())
        open_ops = [i for i in range(start, end) if is_open[i].item()]
        for idx in range(len(open_ops) - 1):
            op_curr = open_ops[idx]
            op_next = open_ops[idx + 1]
            if op_curr in kept_set:
                succ_total += 1
                if op_next in kept_set:
                    succ_kept += 1
    successor_retention = (succ_kept / succ_total) if succ_total > 0 else float("nan")

    # --- 6. Mean frontier distance: kept open vs all open ---
    kept_open = kept_mask & is_open
    mean_frontier_dist_kept    = frontier_dist[kept_open].mean().item() if kept_open.any() else float("nan")
    mean_frontier_dist_all     = frontier_dist[is_open].mean().item()   if is_open.any()   else float("nan")
    kept_dr_mask = kept_mask & is_decision_relevant
    mean_frontier_dist_kept_dr = frontier_dist[kept_dr_mask].mean().item() if kept_dr_mask.any()         else float("nan")
    mean_frontier_dist_all_dr  = frontier_dist[is_decision_relevant].mean().item() if is_decision_relevant.any() else float("nan")

    # --- 7. Kept composition ---
    n_kept_open_other = int((kept_mask & is_open & ~is_eligible & ~is_critical).sum().item())
    n_kept_completed  = int((kept_mask & is_completed).sum().item())
    n_kept_padding    = int((kept_mask & is_padding).sum().item())

    return {
        # Debug counts
        "k":                    k,
        "n_open":               int(is_open.sum().item()),
        "n_eligible":           n_eligible,
        "n_critical":           n_critical,
        "n_decision_relevant":  int(is_decision_relevant.sum().item()),
        "n_kept":               top_idx.shape[0],
        "n_kept_eligible":      n_kept_eligible,
        "n_kept_critical":      n_kept_critical,
        "n_kept_open_other":    n_kept_open_other,
        "n_kept_completed":     n_kept_completed,
        "n_kept_padding":       n_kept_padding,
        # Main metrics
        "slack_correlation":            slack_corr,
        "mean_slack_kept":              mean_slack_kept,
        "mean_slack_discarded":         mean_slack_disc,
        "slack_diff_kept_minus_disc":   slack_diff_kept_minus_disc,
        "critical_retention":           critical_retention,
        "eligible_retention":           eligible_retention,
        "successor_retention":          successor_retention,
        "mean_frontier_dist_kept":      mean_frontier_dist_kept,
        "mean_frontier_dist_all":       mean_frontier_dist_all,
        "mean_frontier_dist_kept_dr":   mean_frontier_dist_kept_dr,
        "mean_frontier_dist_all_dr":    mean_frontier_dist_all_dr,
    }


def compute_random_baseline_metrics(cats, k, env, batch_idx=0, n_samples=10):
    """
    Compute pooling metrics for random node selection as a baseline.
    Respects the same constraints as learned pooling: eligible nodes are always
    kept, padding and completed nodes are excluded from random draws.
    Averages over n_samples draws.
    """
    N            = cats["N"]
    is_eligible  = cats["is_eligible"]
    is_completed = cats["is_completed"]
    is_padding   = cats["is_padding"]

    selectable            = ~is_padding & ~is_completed
    eligible_indices      = torch.where(is_eligible)[0]
    non_elig_selectable   = torch.where(selectable & ~is_eligible)[0]

    n_eligible = eligible_indices.shape[0]
    n_extra = max(0, min(k - n_eligible, non_elig_selectable.shape[0]))

    accumulators = {}

    for _ in range(n_samples):
        perm       = torch.randperm(non_elig_selectable.shape[0])[:n_extra]
        random_extra = non_elig_selectable[perm]
        top_idx, _ = torch.sort(torch.cat([eligible_indices, random_extra]))

        fake_diag = {
            "gate_scores_raw": torch.rand(N).unsqueeze(0),
            "top_idx":         top_idx.unsqueeze(0),
            "k":               k,
        }
        metrics = compute_step_metrics(fake_diag, cats, env, batch_idx)

        for key, val in metrics.items():
            if isinstance(val, (int, float)):
                accumulators.setdefault(key, []).append(val)

    result = {}
    for key, vals in accumulators.items():
        valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
        result[key] = float(np.mean(valid)) if valid else float("nan")

    return result
