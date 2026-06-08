from __future__ import annotations

import math

import torch


def _validate_tau_params(tau_low: float, tau_high: float) -> None:
    if not math.isfinite(tau_low) or not math.isfinite(tau_high):
        raise ValueError("tau bounds must be finite numbers")
    if tau_high <= tau_low:
        raise ValueError("tau_high must be greater than tau_low")


def tbss_band_mask(scores: torch.Tensor, tau_low: float, tau_high: float) -> torch.Tensor:
    """Return mask selecting scores strictly inside the TBSS band."""
    _validate_tau_params(tau_low, tau_high)
    return (scores > tau_low) & (scores < tau_high)


def tbss_weights(
    scores: torch.Tensor,
    tau_low: float,
    tau_high: float,
    lambda_max: float,
) -> torch.Tensor:
    """Compute interpolation weights for TBSS heads.

    Args:
        scores: Tensor of preserve scores with shape [..., tokens].
        tau_low: Lower threshold of the boundary band (must be < tau_high).
        tau_high: Upper threshold of the boundary band.
        lambda_max: Peak adjustment strength at the band midpoint (must be >= 0).
    """
    _validate_tau_params(tau_low, tau_high)
    if lambda_max < 0:
        raise ValueError("lambda_max must be non-negative")



def apply_tbss(
    values: torch.Tensor,
    refs: torch.Tensor,
    scores: torch.Tensor,
    tau_low: float,
    tau_high: float,
    lambda_max: float,
) -> torch.Tensor:
    """Refine TBSS heads in `values` toward `refs` using computed weights.

    `values` and `refs` must share the same shape, and their prefix matching the
    `scores` shape must be identical; the remaining trailing dimensions are
    treated as feature axes to be broadcasted.
    """
