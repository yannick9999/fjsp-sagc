import torch
import torch.nn as nn
import torch.nn.functional as F

from .hgnn import GATedge, MLPsim
from pooling.topk import TopKPool, TopKUnpool


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
    def __init__(self, in_size_ope, in_size_ma, out_size_ope, out_size_ma,
                 hidden_size_ope, num_head, dropout, ratio):
        super().__init__()
        d_ope = out_size_ope
        d_ma = out_size_ma

        self.enc = GCNBlock(in_size_ope, in_size_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)
        self.pool = TopKPool(d_ope, ratio)
        self.btn = GCNBlock(d_ope, d_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)
        self.unpool = TopKUnpool()
        self.dec = GCNBlock(d_ope, d_ma, d_ope, d_ma,
                            hidden_size_ope, num_head, dropout)

    def forward(self, raw_opes, raw_mas, proc_time,
                ope_ma_adj, ope_pre_adj, ope_sub_adj,
                nums_opes, eligible_opes):
        '''
        All inputs already indexed to the active batch.
            raw_opes    [B, N, in_size_ope]
            raw_mas     [B, M, in_size_ma]
            proc_time   [B, N, M]
            ope_ma_adj  [B, N, M]
            ope_pre_adj [B, N, N]
            ope_sub_adj [B, N, N]
            nums_opes   [B]
            eligible_opes [B, N] bool, True = must not be pooled
        Returns:
            h_opes [B, N, out_size_ope]
            h_mas  [B, M, out_size_ma]
        '''
        B = raw_opes.size(0)
        batch_idxes = torch.arange(B, device=raw_opes.device)

        # encoder
        h_opes, h_mas = self.enc(raw_opes, raw_mas, proc_time,
                                 ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes)

        # skip connection, save state from before pooling
        skip_h = h_opes
        skip_pre, skip_sub = ope_pre_adj, ope_sub_adj
        skip_ma, skip_proc = ope_ma_adj, proc_time

        # pool operations, machines untouched
        h_opes_p, pre_p, sub_p, ma_p, proc_p, info = self.pool(
            h_opes, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time,
            nums_opes, eligible_opes)

        # bottleneck at coarse resolution, machine embedding recomputed here
        h_opes_b, h_mas_b = self.btn(h_opes_p, h_mas, proc_p,
                                     ma_p, pre_p, sub_p, batch_idxes)

        # unpool back to full operation count, add skip
        h_opes_u = self.unpool(h_opes_b, info, skip_h)

        # decoder at full resolution with restored adjacencies
        h_opes_out, h_mas_out = self.dec(h_opes_u, h_mas_b, skip_proc,
                                         skip_ma, skip_pre, skip_sub, batch_idxes)

        return h_opes_out, h_mas_out
