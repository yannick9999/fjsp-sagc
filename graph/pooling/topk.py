import torch
import torch.nn as nn
from .mask import build_score_mask


class TopKPool(nn.Module):
    """
    Graph pooling layer based on Gao & Ji (2019) gPool.
    Selects the top-k nodes based on a learned projection vector p.
    """

    def __init__(self, in_feats: int, ratio: float, k_mode: str = "jobs"):
        """
        Inputs:
            in_feats (int):   input feature dimension d
            ratio    (float): fraction of nodes to keep
            k_mode   (str):   'jobs' or 'ops'
        """
        super().__init__()
        self.ratio = ratio
        self.k_mode = k_mode
        self.proj = nn.Linear(in_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def _build_chain_adj(self, top_idx, opes_appertain):
        """
        Rebuild predecessor and successor adjacency by connecting consecutive
        kept operations that belong to the same job.

        Inputs:
            top_idx        (Tensor): kept indices sorted ascending, shape [B, k]
            opes_appertain (Tensor): job index per operation, shape [B, N]

        Outputs:
            pre_adj (Tensor): shape [B, k, k], pre_adj[b, i, j] = 1 iff position j
                              is the immediate kept predecessor of position i
                              within the same job
            sub_adj (Tensor): transpose of pre_adj
        """
        B, k = top_idx.shape
        device = top_idx.device

        # job id of each kept operation
        kept_jobs = opes_appertain.gather(1, top_idx)          # [B, k]

        # consecutive kept positions in same job
        same_job = (kept_jobs[:, 1:] == kept_jobs[:, :-1])     # [B, k-1]

        # edge i -> i+1 in the sorted kept order
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
            ope_pre_adj    (Tensor): predecessor adjacency, directed, shape [B, N, N]
            ope_sub_adj    (Tensor): successor adjacency, directed, shape [B, N, N]
            ope_ma_adj     (Tensor): operation-machine adjacency, bipartite, shape [B, N, M]
            proc_time      (Tensor): edge features, shape [B, N, M]
            nums_opes      (Tensor): number of real operations per instance, shape [B]
            opes_appertain (Tensor): job index per operation, shape [B, N]
            eligible_opes  (Tensor): optional bool mask, shape [B, N], True = node must not be pooled
            completed_opes (Tensor): optional bool mask, shape [B, N], True = completed, exclude from pooling

        Outputs:
            h_pooled      (Tensor): pooled embeddings, shape [B, k, d]
            pre_pooled    (Tensor): pooled predecessor adjacency via chain reconstruction, shape [B, k, k]
            sub_pooled    (Tensor): pooled successor adjacency via chain reconstruction, shape [B, k, k]
            ope_ma_pooled (Tensor): pooled operation-machine adjacency, row reduction only, shape [B, k, M]
            proc_pooled   (Tensor): pooled edge features, row reduction only, shape [B, k, M]
            pool_info     (dict):   keys: top_idx, gate, original_size, nums_opes_pooled,
                                    opes_appertain_pooled, eligible_opes_pooled, completed_opes_pooled
        """
        B, N, d = h.shape

        # raw sigmoid scores are the gate values, they stay in (0, 1)
        gate_scores = torch.sigmoid(self.proj(h).squeeze(-1))

        # mask padding nodes (-inf) and protect eligible nodes (+inf)
        pad_mask, protect_mask = build_score_mask(N, nums_opes, h.device, eligible_opes, completed_opes)
        sel_scores = gate_scores.masked_fill(pad_mask, float('-inf'))
        sel_scores = sel_scores.masked_fill(protect_mask, float('inf'))

        # k must be large enough to include all protected nodes
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

        top_idx = torch.topk(sel_scores, k, dim=-1).indices   # [B, k]
        top_idx, _ = torch.sort(top_idx, dim=-1)           # preserve position order

        # gate comes from the raw sigmoid, never from the masked scores
        gate = gate_scores.gather(1, top_idx)

        # gather selected node embeddings and scale by gate
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)  # [B, k, d]
        h_pooled = h.gather(1, idx_expand)                     # [B, k, d]
        h_pooled = h_pooled * gate.unsqueeze(-1)

        def _pool_rows(mat):
            # bipartite matrices only need row reduction, columns (machines) are unchanged
            M = mat.shape[-1]
            idx_row = top_idx.unsqueeze(-1).expand(-1, -1, M)  # [B, k, M]
            return mat.gather(1, idx_row)                       # [B, k, M]

        pre_pooled, sub_pooled = self._build_chain_adj(top_idx, opes_appertain)
        ope_ma_pooled = _pool_rows(ope_ma_adj)
        proc_pooled = _pool_rows(proc_time)

        nums_opes_pooled = torch.full((B,), k, dtype=nums_opes.dtype, device=h.device)
        opes_appertain_pooled = opes_appertain.gather(1, top_idx)
        if eligible_opes is not None:
            eligible_opes_pooled = eligible_opes.gather(1, top_idx)
        else:
            eligible_opes_pooled = None

        if completed_opes is not None:
            completed_opes_pooled = completed_opes.gather(1, top_idx)
        else:
            completed_opes_pooled = None

        pool_info = {
            "top_idx":                top_idx,                # [B, k]
            "gate":                   gate,                   # [B, k]
            "original_size":          N,
            "nums_opes_pooled":       nums_opes_pooled,       # [B]
            "opes_appertain_pooled":  opes_appertain_pooled,  # [B, k]
            "eligible_opes_pooled":   eligible_opes_pooled,   # [B, k] or None
            "completed_opes_pooled":  completed_opes_pooled,  # [B, k] or None
        }

        return h_pooled, pre_pooled, sub_pooled, ope_ma_pooled, proc_pooled, pool_info


class TopKUnpool(nn.Module):
    """
    Graph unpooling layer matching TopKPool.
    Reconstructs the original node count by scattering pooled
    embeddings back to their original positions, then adds the
    skip connection from before pooling.
    """

    def forward(self, h_pooled, pool_info, skip_h):
        """
        Inputs:
            h_pooled    (Tensor): embeddings from bottleneck, shape [B, k, d]
            pool_info   (dict):   saved info from TopKPool
            skip_h      (Tensor): embeddings from before pooling, shape [B, N, d]

        Outputs:
            h_out        (Tensor): reconstructed embeddings, shape [B, N, d]
        """
        top_idx       = pool_info["top_idx"]        # [B, k]
        original_size = pool_info["original_size"]  # N

        B, k, d = h_pooled.shape

        # scatter pooled embeddings back to their original positions
        h_out = torch.zeros(B, original_size, d, device=h_pooled.device)
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)  # [B, k, d]
        h_out.scatter_(1, idx_expand, h_pooled)

        # add skip connection from before pooling
        h_out = h_out + skip_h

        return h_out
