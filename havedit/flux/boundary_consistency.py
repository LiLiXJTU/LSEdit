from __future__ import annotations

import math

import torch


def _validate_tau_params(tau_low: float, tau_high: float) -> None:
    if not math.isfinite(tau_low) or not math.isfinite(tau_high):
        raise ValueError("tau bounds must be finite numbers")
    if tau_high <= tau_low:
        raise ValueError("tau_high must be greater than tau_low")


def boundary_band_mask(scores: torch.Tensor, tau_low: float, tau_high: float) -> torch.Tensor:
    """Return mask selecting scores strictly inside the boundary band."""
    _validate_tau_params(tau_low, tau_high)
    return (scores > tau_low) & (scores < tau_high)


def boundary_consistency_weights(
    scores: torch.Tensor,
    tau_low: float,
    tau_high: float,
    lambda_max: float,
) -> torch.Tensor:
    """Compute interpolation weights for boundary heads.

    Args:
        scores: Tensor of preserve scores with shape [..., tokens].
        tau_low: Lower threshold of the boundary band (must be < tau_high).
        tau_high: Upper threshold of the boundary band.
        lambda_max: Peak adjustment strength at the band midpoint (must be >= 0).
    """
    _validate_tau_params(tau_low, tau_high)
    if lambda_max < 0:
        raise ValueError("lambda_max must be non-negative")

    center = (tau_low + tau_high) / 2.0
    half_width = (tau_high - tau_low) / 2.0
    ramp = 1.0 - (scores - center).abs() / half_width
    tapered = ramp.clamp(min=0.0, max=1.0)
    return lambda_max * tapered


def apply_boundary_head_consistency(
    values: torch.Tensor,
    refs: torch.Tensor,
    scores: torch.Tensor,
    tau_low: float,
    tau_high: float,
    lambda_max: float,
) -> torch.Tensor:
    """Refine boundary heads in `values` toward `refs` using computed weights.

    `values` and `refs` must share the same shape, and their prefix matching the
    `scores` shape must be identical; the remaining trailing dimensions are
    treated as feature axes to be broadcasted.
    """
    weights = boundary_consistency_weights(
        scores,
        tau_low=tau_low,
        tau_high=tau_high,
        lambda_max=lambda_max,
    )
    weights = weights.clamp(min=0.0, max=1.0)
    if values.shape != refs.shape:
        raise ValueError("values and refs must share the same shape")
    if values.shape[: scores.dim()] != scores.shape:
        raise ValueError("scores shape must match the leading dimensions of values and refs")

    if values.device != refs.device or values.device != scores.device:
        raise ValueError("values, refs, and scores must be on the same device")

    feature_dims = values.dim() - scores.dim()
    expanded_weights = weights.reshape(*weights.shape, *([1] * feature_dims))
    return torch.lerp(values, refs, expanded_weights.to(dtype=values.dtype))
