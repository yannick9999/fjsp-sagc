import torch
import torch.nn as nn
from .mask import build_score_mask


class RandomPool(nn.Module):
    """
    Random pooling: selects k nodes uniformly at random per instance.
    Respects the same constraints as SAGC: padding and completed nodes
    are excluded, eligible nodes are protected (always kept).
    No learned parameters. Serves as a baseline for diagnostics.
    """

    def __init__(self, in_feats: int, ratio: float, k_mode: str = "jobs"):
        super().__init__()
        self.ratio = ratio
        self.k_mode = k_mode
        self._dummy = nn.Parameter(torch.zeros(1))
        self.diagnostic_mode = False
        self._last_diag = None

    def _build_chain_adj(self, top_idx, opes_appertain):
        B, k = top_idx.shape
        device = top_idx.device
        kept_jobs = opes_appertain.gather(1, top_idx)
        same_job = (kept_jobs[:, 1:] == kept_jobs[:, :-1])
        pre_adj = torch.zeros(B, k, k, device=device)
        rows = torch.arange(k - 1, device=device)
        cols = rows + 1
        pre_adj[:, rows, cols] = same_job.float()
        sub_adj = pre_adj.transpose(1, 2).contiguous()
        return pre_adj, sub_adj

    def forward(self, h, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time,
                nums_opes, opes_appertain, eligible_opes=None, completed_opes=None,
                ope_feats=None):
        B, N, d = h.shape
        device = h.device

        gate_scores = torch.rand(B, N, device=device)

        pad_mask, protect_mask = build_score_mask(N, nums_opes, device,
                                                   eligible_opes, completed_opes)
        sel_scores = gate_scores.masked_fill(pad_mask, float('-inf'))
        sel_scores = sel_scores.masked_fill(protect_mask, float('inf'))

        num_protected = int(protect_mask.sum(dim=-1).max().item())

        if self.k_mode == "jobs":
            num_jobs = int(opes_appertain.max().item()) + 1
            k_target = max(1, int(self.ratio * num_jobs))
        elif self.k_mode == "ops":
            k_target = max(1, int(self.ratio * nums_opes.min().item()))
        else:
            raise ValueError(f"Unknown k_mode '{self.k_mode}'.")

        k = max(num_protected, k_target)
        k = min(k, int(nums_opes.min().item()))

        top_idx = torch.topk(sel_scores, k, dim=-1).indices
        top_idx, _ = torch.sort(top_idx, dim=-1)

        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)
        h_pooled = h.gather(1, idx_expand)

        def _pool_rows(mat):
            M = mat.shape[-1]
            idx_row = top_idx.unsqueeze(-1).expand(-1, -1, M)
            return mat.gather(1, idx_row)

        pre_pooled, sub_pooled = self._build_chain_adj(top_idx, opes_appertain)
        ope_ma_pooled = _pool_rows(ope_ma_adj)
        proc_pooled = _pool_rows(proc_time)

        nums_opes_pooled = torch.full((B,), k, dtype=nums_opes.dtype, device=device)
        opes_appertain_pooled = opes_appertain.gather(1, top_idx)
        eligible_opes_pooled = eligible_opes.gather(1, top_idx) if eligible_opes is not None else None
        completed_opes_pooled = completed_opes.gather(1, top_idx) if completed_opes is not None else None

        if ope_feats is not None:
            f = ope_feats.shape[-1]
            idx_feats = top_idx.unsqueeze(-1).expand(-1, -1, f)
            ope_feats_pooled = ope_feats.gather(1, idx_feats)
        else:
            ope_feats_pooled = None

        pool_info = {
            "top_idx":                top_idx,
            "gate":                   None,
            "original_size":          N,
            "nums_opes_pooled":       nums_opes_pooled,
            "opes_appertain_pooled":  opes_appertain_pooled,
            "eligible_opes_pooled":   eligible_opes_pooled,
            "completed_opes_pooled":  completed_opes_pooled,
            "ope_feats_pooled":       ope_feats_pooled,
        }

        if self.diagnostic_mode:
            self._last_diag = {
                "gate_scores_raw": gate_scores.detach().cpu(),
                "sel_scores": sel_scores.detach().cpu(),
                "top_idx": top_idx.detach().cpu(),
                "k": k,
            }

        return h_pooled, pre_pooled, sub_pooled, ope_ma_pooled, proc_pooled, pool_info
