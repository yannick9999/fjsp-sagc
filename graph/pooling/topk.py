import torch
import torch.nn as nn
from .mask import build_score_mask


class TopKPool(nn.Module):
    """
    Graph pooling layer based on Gao & Ji (2019) gPool.
    Selects the top-k nodes based on a learned projection vector p.
    """

    def __init__(self, in_feats: int, ratio: float):
        """
        Inputs:
            in_feats (int):   input feature dimension d
            ratio    (float): fraction of nodes to keep
        """
        super().__init__()
        self.ratio = ratio
        self.proj = nn.Linear(in_feats, 1, bias=False)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, h, adj, nums_opes, eligible_opes=None):
        """
        Inputs:
            h             (Tensor): node embeddings, shape [B, N, d]
            adj           (Tensor): adjacency matrix, shape [B, N, N]
            nums_opes     (Tensor): number of real operations per instance, shape [B]
            eligible_opes (Tensor): optional bool mask, shape [B, N], True = node must not be pooled

        Outputs:
            h_pooled   (Tensor): pooled embeddings, shape [B, k, d]
            adj_pooled (Tensor): pooled adjacency matrix with graph power, shape [B, k, k]
            pool_info  (dict):   keys: top_idx, gate, original_size
        """
        B, N, d = h.shape

        # project each node to a scalar score and apply sigmoid
        scores = self.proj(h).squeeze(-1)
        scores = torch.sigmoid(scores)

        # mask padding nodes (-inf) and protect eligible nodes (+inf)
        pad_mask, protect_mask = build_score_mask(N, nums_opes, h.device, eligible_opes)
        scores = scores.masked_fill(pad_mask, float('-inf'))
        scores = scores.masked_fill(protect_mask, float('inf'))

        # k must be large enough to include all protected nodes
        num_protected = protect_mask.sum(dim=-1).max().item()
        k = max(int(num_protected), max(1, int(self.ratio * nums_opes.min().item())))
        top_idx = torch.topk(scores, k, dim=-1).indices   # [B, k]
        top_idx, _ = torch.sort(top_idx, dim=-1)           # preserve position order

        # gate scores of selected nodes; scores are already sigmoid values
        gate = scores.gather(1, top_idx)    # [B, k]

        # gather selected node embeddings and scale by gate
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)  # [B, k, d]
        h_pooled = h.gather(1, idx_expand)                     # [B, k, d]
        h_pooled = h_pooled * gate.unsqueeze(-1)

        # reduce adjacency matrix to selected nodes by filtering rows then columns
        idx_row = top_idx.unsqueeze(-1).expand(-1, -1, N)      # [B, k, N]
        adj_pooled = adj.gather(1, idx_row)                    # [B, k, N]
        idx_col = top_idx.unsqueeze(1).expand(-1, k, -1)       # [B, k, k]
        adj_pooled = adj_pooled.gather(2, idx_col)             # [B, k, k]

        # A + A^2 reconnects nodes that had a path through a removed node
        adj_sq = torch.bmm(adj_pooled, adj_pooled)
        adj_pooled = (adj_pooled + adj_sq).clamp(0, 1)

        pool_info = {
            "top_idx":       top_idx,   # [B, k]
            "gate":          gate,      # [B, k]
            "original_size": N,
        }

        return h_pooled, adj_pooled, pool_info


class TopKUnpool(nn.Module):
    """
    Graph unpooling layer matching TopKPool.
    Reconstructs the original node count by scattering pooled
    embeddings back to their original positions, then adds the
    skip connection from before pooling.
    """

    def forward(self, h_pooled, pool_info, skip_h, original_adj):
        """
        Inputs:
            h_pooled    (Tensor): embeddings from bottleneck, shape [B, k, d]
            pool_info   (dict):   saved info from TopKPool
            skip_h      (Tensor): embeddings from before pooling, shape [B, N, d]
            original_adj(Tensor): adjacency matrix from before pooling, shape [B, N, N]

        Outputs:
            h_out        (Tensor): reconstructed embeddings, shape [B, N, d]
            original_adj (Tensor): passed through unchanged
        """
        top_idx       = pool_info["top_idx"]        # [B, k]
        gate          = pool_info["gate"]            # [B, k]
        original_size = pool_info["original_size"]  # N

        B, k, d = h_pooled.shape

        # re-apply gate on the way back, matching Gao & Ji (2019)
        h_scattered = h_pooled * gate.unsqueeze(-1)   # [B, k, d]

        # scatter pooled embeddings back to their original positions
        h_out = torch.zeros(B, original_size, d, device=h_pooled.device)
        idx_expand = top_idx.unsqueeze(-1).expand(-1, -1, d)  # [B, k, d]
        h_out.scatter_(1, idx_expand, h_scattered)

        # add skip connection from before pooling
        h_out = h_out + skip_h

        return h_out, original_adj
