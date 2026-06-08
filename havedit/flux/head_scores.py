from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_head_deviation(v_pred: torch.Tensor, v_ref: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(v_pred - v_ref, dim=-1)


def _local_zscore_map(
    deviation: torch.Tensor,
    height: int,
    width: int,
    kernel_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    batch, heads, tokens = deviation.shape
    x = deviation.reshape(batch * heads, 1, height, width)
    mean = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    mean_sq = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    var = (mean_sq - mean * mean).clamp_min(0)
    std = var.sqrt()
    return ((x - mean) / (std + eps)).reshape(batch, heads, tokens)


def compute_preserve_scores(
    deviation: torch.Tensor,
    semantic_prior: torch.Tensor,
    height: int,
    width: int,
    alpha: float,
    beta: float,
    kernel_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    z = _local_zscore_map(deviation, height=height, width=width, kernel_size=kernel_size, eps=eps)
    sem = semantic_prior.reshape(semantic_prior.shape[0], 1, -1).to(dtype=z.dtype, device=z.device)
    return torch.sigmoid(-alpha * z + beta * (1 - sem))


def replace_values_from_scores(
    v_pred: torch.Tensor,
    v_ref: torch.Tensor,
    scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    return torch.where((scores > threshold).unsqueeze(-1), v_ref, v_pred)


def reduce_scores_to_token_mask(scores: torch.Tensor, threshold: float) -> torch.Tensor:
    return scores.mean(dim=1) > threshold


def expand_token_mask_to_head_mask(token_mask: torch.Tensor, num_heads: int) -> torch.Tensor:
    return token_mask.unsqueeze(1).expand(-1, num_heads, -1)
