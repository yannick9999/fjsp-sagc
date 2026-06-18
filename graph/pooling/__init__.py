from .topk import TopKPool, TopKUnpool
from .sagc import SAGCPool
from .identity import IdentityPool, IdentityUnpool


POOLING_REGISTRY = {
    "topk": (TopKPool, TopKUnpool),
    "sagc": (SAGCPool, TopKUnpool),
    "nopooling": (IdentityPool, IdentityUnpool),
}


def build_pooling(model_paras: dict):
    """
    Build a pooling layer from model_paras.

    Inputs:
        model_paras (dict): full model_paras dict, must contain
                            'pooling' sub-dict and 'out_size_ope'

    Outputs:
        TopKPool instance (or matching pool class)
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

    pool_cls, _ = POOLING_REGISTRY[method]

    if method == "nopooling":
        return pool_cls()
    elif method == "sagc":
        ope_feat_dim = model_paras["in_size_ope"]
        return pool_cls(in_feats, ope_feat_dim, ratio, k_mode)
    else:
        return pool_cls(in_feats, ratio, k_mode)


def build_unpooling(model_paras: dict):
    """
    Build an unpooling layer matching the pooling config.

    Inputs:
        model_paras (dict): full model_paras dict, must contain
                            'pooling' sub-dict

    Outputs:
        TopKUnpool instance (or matching unpool class)
    """
    cfg    = model_paras["pooling"]
    method = cfg["method"]

    if method not in POOLING_REGISTRY:
        raise ValueError(
            f"Unknown pooling method '{method}'. "
            f"Available: {list(POOLING_REGISTRY.keys())}"
        )

    _, unpool_cls = POOLING_REGISTRY[method]
    return unpool_cls()
