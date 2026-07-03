from .topk import TopKPool
from .sagc import SAGCPool
from .identity import IdentityPool


POOLING_REGISTRY = {
    "topk": TopKPool,
    "sagc": SAGCPool,
    "nopooling": IdentityPool,
}


def build_pooling(model_paras: dict):
    """
    Build a pooling layer from model_paras.

    Inputs:
        model_paras (dict): full model_paras dict, must contain
                            'pooling' sub-dict and 'out_size_ope'

    Outputs:
        Pool instance (TopKPool, SAGCPool, or IdentityPool)
    """
    cfg    = model_paras["pooling"]
    method = cfg["method"]

    if method not in POOLING_REGISTRY:
        raise ValueError(
            f"Unknown pooling method '{method}'. "
            f"Available: {list(POOLING_REGISTRY.keys())}"
        )

    in_feats = model_paras["out_size_ope"]
    ratio    = cfg["ratio"]
    k_mode   = cfg.get("k_mode", "jobs")

    pool_cls = POOLING_REGISTRY[method]

    if method == "nopooling":
        return pool_cls()
    elif method == "sagc":
        ope_feat_dim = model_paras["in_size_ope"]
        return pool_cls(in_feats, ope_feat_dim, ratio, k_mode)
    else:
        return pool_cls(in_feats, ratio, k_mode)
