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
    Variable-depth Graph U-Net for the FJSP operation graph.
    Structure for num_pool_layers = L:
        gcn_0 -> pool_1 -> gcn_1 -> ... -> pool_L -> gcn_L (bottleneck)
                                                           |
        gcn_2L <- unpool_1 <- gcn_2L-1 <- ... <- unpool_L <+
    Returns full-size operation embeddings and machine embeddings at full resolution.
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
        num_pool_layers = model_paras["pooling"].get("num_layers", 1)
        self.num_pool_layers = num_pool_layers

        self.gcns = nn.ModuleList()
        for i in range(2 * num_pool_layers + 1):
            if i == 0:
                self.gcns.append(GCNBlock(in_size_ope, in_size_ma, d_ope, d_ma,
                                          hidden_size_ope, num_head, dropout))
            else:
                self.gcns.append(GCNBlock(d_ope, d_ma, d_ope, d_ma,
                                          hidden_size_ope, num_head, dropout))

        self.pools   = nn.ModuleList([build_pooling(model_paras)   for _ in range(num_pool_layers)])
        self.unpools = nn.ModuleList([build_unpooling(model_paras) for _ in range(num_pool_layers)])

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
                nums_opes, opes_appertain, eligible_opes, completed_opes=None):
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
            completed_opes [B, N] bool, True = completed, exclude from pooling
        Returns:
            h_opes [B, N, out_size_ope]
            h_mas  [B, M, out_size_ma]
        '''
        B = raw_opes.size(0)
        batch_idxes = torch.arange(B, device=raw_opes.device)
        L = self.num_pool_layers
        _is_cuda = raw_opes.device.type == 'cuda'

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            _t_fwd_start = time.perf_counter()
            _t_coarse_acc = 0.0

        h_opes, h_mas = self.gcns[0](raw_opes, raw_mas, proc_time,
                                     ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes)

        # down path: push skip, pool, GCN
        skips = []
        cur_pre, cur_sub = ope_pre_adj, ope_sub_adj
        cur_ma,  cur_proc = ope_ma_adj, proc_time
        cur_nums_opes = nums_opes
        cur_appertain = opes_appertain
        cur_eligible  = eligible_opes
        cur_completed = completed_opes

        for i in range(L):
            skips.append({"h": h_opes, "pre": cur_pre, "sub": cur_sub,
                          "ma": cur_ma, "proc": cur_proc})

            if self._timing_enabled:
                if _is_cuda:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()

            h_opes, cur_pre, cur_sub, cur_ma, cur_proc, info = self.pools[i](
                h_opes, cur_pre, cur_sub, cur_ma, cur_proc,
                cur_nums_opes, cur_appertain, cur_eligible, cur_completed)

            if self._timing_enabled:
                if _is_cuda:
                    torch.cuda.synchronize()
                _t_coarse_acc += time.perf_counter() - _t0

            cur_nums_opes = info["nums_opes_pooled"]
            cur_appertain = info["opes_appertain_pooled"]
            cur_eligible  = info["eligible_opes_pooled"]
            cur_completed = info["completed_opes_pooled"]
            skips[-1]["info"] = info

            h_opes, h_mas = self.gcns[i + 1](h_opes, h_mas, cur_proc,
                                              cur_ma, cur_pre, cur_sub, batch_idxes)

        # up path: pop skip, unpool, GCN
        for i in range(L):
            skip = skips.pop()

            if self._timing_enabled:
                if _is_cuda:
                    torch.cuda.synchronize()
                _t0 = time.perf_counter()

            h_opes = self.unpools[i](h_opes, skip["info"], skip["h"])

            if self._timing_enabled:
                if _is_cuda:
                    torch.cuda.synchronize()
                _t_coarse_acc += time.perf_counter() - _t0

            cur_pre, cur_sub = skip["pre"], skip["sub"]
            cur_ma,  cur_proc = skip["ma"],  skip["proc"]
            h_opes, h_mas = self.gcns[L + 1 + i](h_opes, h_mas, cur_proc,
                                                  cur_ma, cur_pre, cur_sub, batch_idxes)

        if self._timing_enabled:
            if _is_cuda:
                torch.cuda.synchronize()
            self._t_coarse_total  += _t_coarse_acc
            self._t_forward_total += time.perf_counter() - _t_fwd_start
            self._n_calls += 1

        return h_opes, h_mas
