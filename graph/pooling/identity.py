import torch.nn as nn


class IdentityPool(nn.Module):

    def forward(self, h, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time,
                nums_opes, opes_appertain, eligible_opes=None, completed_opes=None,
                ope_feats=None):
        B, N, d = h.shape

        pool_info = {
            "top_idx": None,
            "gate": None,
            "original_size": N,
            "nums_opes_pooled": nums_opes,
            "opes_appertain_pooled": opes_appertain,
            "eligible_opes_pooled": eligible_opes,
            "completed_opes_pooled": completed_opes,
        }
        if ope_feats is not None:
            pool_info["ope_feats_pooled"] = ope_feats

        return h, ope_pre_adj, ope_sub_adj, ope_ma_adj, proc_time, pool_info


class IdentityUnpool(nn.Module):

    def forward(self, h_pooled, pool_info, skip_h):
        return h_pooled + skip_h
