import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgnn import GATedge, MLPsim
from .pooling import build_pooling, build_unpooling


class MLPs(nn.Module):
    '''
    Operation node embedding, identical to Song's MLPs.
    Kept here so a GCNBlock is self-contained. Five MLPsim aggregate
    information from machines, predecessors, successors and self,
    then a projection fuses them.
    '''
    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        super().__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope,
                                          self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, feats):
        # feats = (h_opes, h_mas, proc_time)
        h = (feats[1], feats[0], feats[0], feats[0])
        self_adj = torch.eye(feats[0].size(-2), dtype=torch.int64,
                             device=feats[0].device).unsqueeze(0).expand_as(ope_pre_adj)
        adj = (ope_ma_adj, ope_pre_adj, ope_sub_adj, self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime


class GCNBlock(nn.Module):
    '''
    One HGNN layer in Song's sense, packaged as a single building block
    for the Graph U-Net. Stage 1 computes machine embeddings with GATedge,
    stage 2 computes operation embeddings with the MLPs.
    '''
    def __init__(self, in_size_ope, in_size_ma, out_size_ope, out_size_ma,
                 hidden_size_ope, num_head, dropout):
        super().__init__()
        self.gat = GATedge((in_size_ope, in_size_ma), out_size_ma, num_head,
                           dropout, dropout, activation=F.elu)
        self.mlps = MLPs([out_size_ma, in_size_ope, in_size_ope, in_size_ope],
                         hidden_size_ope, out_size_ope, num_head, dropout)

    def forward(self, h_opes, h_mas, proc_time,
                ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes):
        '''
        Inputs (all already indexed to the active batch):
            h_opes      [B, n, in_size_ope]
            h_mas       [B, M, in_size_ma]
            proc_time   [B, n, M]
            ope_ma_adj  [B, n, M]
            ope_pre_adj [B, n, n]
            ope_sub_adj [B, n, n]
            batch_idxes [B]   typically arange(B)
        Outputs:
            h_opes_out  [B, n, out_size_ope]
            h_mas_out   [B, M, out_size_ma]
        '''
        feats = (h_opes, h_mas, proc_time)
        h_mas_out = self.gat(ope_ma_adj, batch_idxes, feats)
        feats = (h_opes, h_mas_out, proc_time)
        h_opes_out = self.mlps(ope_ma_adj, ope_pre_adj, ope_sub_adj, feats)
        return h_opes_out, h_mas_out


class GraphUNet(nn.Module):
    '''
    Single pooling layer Graph U-Net for the FJSP operation graph.
    Structure: GCN (enc) -> pool -> GCN (bottleneck) -> unpool -> GCN (dec).
    Returns full size operation embeddings so the actor keeps node level
    resolution, plus machine embeddings recomputed at full resolution.
    '''
    def __init__(self, model_paras):
        super().__init__()
        in_size_ope     = model_paras["in_size_ope"]
        in_size_ma      = model_paras["in_size_ma"]
        d_ope           = model_paras["out_size_ope"]
        d_ma            = model_paras["out_size_ma"]
        hidden_size_ope = model_paras["hidden_size_ope"]
        num_head        = model_paras["num_heads"][0]
        dropout         = model_paras["dropout"]

        self.enc = GCNBlock(in_size_ope, in_size_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)
        self.pool = build_pooling(model_paras)
        self.btn = GCNBlock(d_ope, d_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)
        self.unpool = build_unpooling(model_paras)
        self.dec = GCNBlock(d_ope, d_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)

        # coarsening overhead timing, disabled by default so training is unaffected
        self._timing_enabled = False
        self._t_coarse_total = 0.0
        self._t_forward_total = 0.0
        self._n_calls = 0

    def reset_timing(self):
        self._t_coarse_total = 0.0
        self._t_forward_total = 0.0
        self._n_calls = 0

    def get_timing_stats(self):
        if self._n_calls == 0:
            return {"avg_coarse_ms": 0.0, "avg_forward_ms": 0.0, "overhead_pct": 0.0}
        avg_coarse_ms = self._t_coarse_total / self._n_calls * 1000
        avg_forward_ms = self._t_forward_total / self._n_calls * 1000
        overhead_pct = (self._t_coarse_total / self._t_forward_total * 100
                        if self._t_forward_total > 0 else 0.0)
        return {"avg_coarse_ms": avg_coarse_ms, "avg_forward_ms": avg_forward_ms,
                "overhead_pct": overhead_pct}

    def forward(self, raw_opes, raw_mas, proc_time,
                ope_ma_adj, ope_pre_adj, ope_sub_adj,
                nums_opes, opes_appertain, eligible_opes):
        '''
        All inputs already indexed to the active batch.
            raw_opes       [B, N, in_size_ope]
            raw_mas        [B, M, in_size_ma]
            proc_time      [B, N, M]
            ope_ma_adj     [B, N, M]
            ope_pre_adj    [B, N, N]
            ope_sub_adj    [B, N, N]
            nums_opes      [B]
            opes_appertain [B, N] job index per operation
            eligible_opes  [B, N] bool, True = must not be pooled
        Returns:
            h_opes [B, N, out_size_ope]
            h_mas  [B, M, out_size_ma]
        '''
        B = raw_opes.size(0)
        batch_idxes = torch.arange(B, device=raw_opes.device)

        if self._timing_enabled:
            _is_cuda = raw_opes.device.type == 'cuda'
            if _is_cuda:
                torch.cuda.synchronize()
            _t_fwd_start = time.perf_counter()

        # encoder
        h_opes, h_mas = self.enc(raw_opes, raw_mas, proc_time,
                                 ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes)

        # skip connection, save state from before pooling
        skip_h = h_opes
        skip_pre, skip_sub = ope_pre_adj, ope_sub_adj
        skip_ma, skip_proc = ope_ma_adj, proc_time

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_pool_start = time.perf_counter()

        # pool operations, machines untouched
        h_opes_p, pre_p, sub_p, ma_p, proc_p, info = self.pool(
            h_opes, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time,
            nums_opes, opes_appertain, eligible_opes)

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_pool_end = time.perf_counter()

        # bottleneck at coarse resolution, machine embedding recomputed here
        h_opes_b, h_mas_b = self.btn(h_opes_p, h_mas, proc_p,
                                     ma_p, pre_p, sub_p, batch_idxes)

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_unpool_start = time.perf_counter()

        # unpool back to full operation count, add skip
        h_opes_u = self.unpool(h_opes_b, info, skip_h)

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_unpool_end = time.perf_counter()

        # decoder at full resolution with restored adjacencies
        h_opes_out, h_mas_out = self.dec(h_opes_u, h_mas_b, skip_proc,
                                         skip_ma, skip_pre, skip_sub, batch_idxes)

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_fwd_end = time.perf_counter()
            self._t_coarse_total += (_t_pool_end - _t_pool_start) + (_t_unpool_end - _t_unpool_start)
            self._t_forward_total += _t_fwd_end - _t_fwd_start
            self._n_calls += 1

        return h_opes_out, h_mas_out
