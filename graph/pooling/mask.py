import torch


def build_score_mask(N, nums_opes, device, eligible_opes=None, completed_opes=None):
    """
    Inputs:
        N             (int):    total number of nodes (padded)
        nums_opes     (Tensor): real node count per instance, shape [B]
        device:                 torch device
        eligible_opes (Tensor): bool mask of eligible operation nodes,
                                shape [B, N], True = must not be pooled
        completed_opes (Tensor): bool mask of completed operation nodes,
                                 shape [B, N], True = completed, exclude from pooling

    Outputs:
        pad_mask     (Tensor): True where node is padding OR completed, shape [B, N]
        protect_mask (Tensor): True where node must be kept, shape [B, N]
    """
    pad_mask = torch.arange(N, device=device).unsqueeze(0) >= nums_opes.unsqueeze(1)

    if completed_opes is not None:
        pad_mask = pad_mask | completed_opes

    if eligible_opes is not None:
        protect_mask = eligible_opes & ~pad_mask
    else:
        protect_mask = torch.zeros_like(pad_mask)

    return pad_mask, protect_mask
