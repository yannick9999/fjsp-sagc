import torch
import torch.nn as nn
from .mask import build_score_mask


class SAGCPool(nn.Module):
    """
    Scheduling-Aware Graph Coarsening (SAGC) pooling layer.
    Like TopK (Gao & Ji 2019), but the retention score is computed
    from [GNN embedding || normalized operation features] instead of
    the GNN embedding alone.
    """

    def __init__(self, in_feats: int, ope_feat_dim: int, ratio: float, k_mode: str = "jobs"):
        """
        Inputs:
            in_feats     (int):   GNN embedding dimension d (out_size_ope)
            ope_feat_dim (int):   operation feature dimension (in_size_ope)
            ratio        (float): pooling ratio (interpretation depends on k_mode)
            k_mode       (str):   'jobs' or 'ops'
        """
        super().__init__()
        self.ratio = ratio
        self.k_mode = k_mode
        self.proj = nn.Linear(in_feats + ope_feat_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)
        self.diagnostic_mode = False
        self._last_diag = None

    def _build_chain_adj(self, top_idx, opes_appertain):
        B, k = top_idx.shape
        device = top_idx.device
        kept_jobs = opes_appertain.gather(1, top_idx)
        same_job = (kept_jobs[:, 1:] == kept_jobs[:, :-1])
        sub_adj = torch.zeros(B, k, k, device=device)
        rows = torch.arange(k - 1, device=device)
        cols = rows + 1
        sub_adj[:, rows, cols] = same_job.float()
        pre_adj = sub_adj.transpose(1, 2).contiguous()
        return pre_adj, sub_adj

    def forward(self, h, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time,
                nums_opes, opes_appertain, eligible_opes=None, completed_opes=None,
                ope_feats=None):
        """
        Inputs:
            h              (Tensor): node embeddings, shape [B, N, d]
            ope_pre_adj    (Tensor): predecessor adjacency, shape [B, N, N]
            ope_sub_adj    (Tensor): successor adjacency, shape [B, N, N]
            ope_ma_adj     (Tensor): operation-machine adjacency, shape [B, N, M]
            proc_time      (Tensor): edge features, shape [B, N, M]
            nums_opes      (Tensor): number of real operations per instance, shape [B]
            opes_appertain (Tensor): job index per operation, shape [B, N]
            eligible_opes  (Tensor): optional bool mask, shape [B, N], True = must keep
            completed_opes (Tensor): optional bool mask, shape [B, N], True = completed
            ope_feats      (Tensor): normalized operation features, shape [B, N, f]

        Outputs:
            h_pooled        (Tensor): pooled embeddings, shape [B, k, d]
            pre_pooled      (Tensor): pooled predecessor adjacency, shape [B, k, k]
            sub_pooled      (Tensor): pooled successor adjacency, shape [B, k, k]
            ope_ma_pooled   (Tensor): pooled operation-machine adjacency, shape [B, k, M]
            proc_pooled     (Tensor): pooled edge features, shape [B, k, M]
            pool_info       (dict):   includes ope_feats_pooled for multi-level SAGC
        """
        B, N, d = h.shape

        if ope_feats is None:
            raise ValueError("SAGCPool requires ope_feats (normalized operation features)")

        score_input = torch.cat([h, ope_feats], dim=-1)  # [B, N, d + f]
        gate_scores = torch.sigmoid(self.proj(score_input).squeeze(-1))  # [B, N]

        pad_mask, protect_mask = build_score_mask(N, nums_opes, h.device,
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
            raise ValueError(f"Unknown k_mode '{self.k_mode}'. Use 'jobs' or 'ops'.")

        k = max(num_protected, k_target)
        k = min(k, int(nums_opes.min().item()))

        top_idx = torch.topk(sel_scores, k, dim=-1).indices
        top_idx, _ = torch.sort(top_idx, dim=-1)

        gate = gate_scores.gather(1, top_idx)

        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)
        h_pooled = h.gather(1, idx_expand)
        h_pooled = h_pooled * gate.unsqueeze(-1)

        def _pool_rows(mat):
            M = mat.shape[-1]
            idx_row = top_idx.unsqueeze(-1).expand(-1, -1, M)
            return mat.gather(1, idx_row)

        pre_pooled, sub_pooled = self._build_chain_adj(top_idx, opes_appertain)
        ope_ma_pooled = _pool_rows(ope_ma_adj)
        proc_pooled = _pool_rows(proc_time)

        nums_opes_pooled = torch.full((B,), k, dtype=nums_opes.dtype, device=h.device)
        opes_appertain_pooled = opes_appertain.gather(1, top_idx)
        eligible_opes_pooled = eligible_opes.gather(1, top_idx) if eligible_opes is not None else None
        completed_opes_pooled = completed_opes.gather(1, top_idx) if completed_opes is not None else None

        f = ope_feats.shape[-1]
        idx_feats = top_idx.unsqueeze(-1).expand(-1, -1, f)
        ope_feats_pooled = ope_feats.gather(1, idx_feats)

        pool_info = {
            "top_idx":                top_idx,
            "gate":                   gate,
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
