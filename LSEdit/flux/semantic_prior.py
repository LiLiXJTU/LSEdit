from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_text_to_latent_block(attn_probs: torch.Tensor, text_n: int, latent_n: int) -> torch.Tensor:
    return attn_probs[:, :, :text_n, text_n : text_n + latent_n]


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    g1 = torch.exp(-(coords**2) / (2 * sigma**2))
    g1 = g1 / g1.sum()
    kernel = torch.outer(g1, g1)
    return kernel.view(1, 1, kernel_size, kernel_size)


class WarmupSemanticPrior:
    def __init__(self, kernel_size: int = 5, sigma: float = 1.0, eps: float = 1e-6):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.eps = eps
        self.accumulator = None
        self.count = 0

    def update(self, attn_probs: torch.Tensor, text_n: int, latent_n: int) -> None:
        block = extract_text_to_latent_block(attn_probs, text_n=text_n, latent_n=latent_n)
        spatial = block.mean(dim=(1, 2))
        self.accumulator = spatial if self.accumulator is None else self.accumulator + spatial
        self.count += 1

    def finalize(self, height: int, width: int) -> torch.Tensor:
        prior = (self.accumulator / max(self.count, 1)).view(-1, 1, height, width)
        kernel = _gaussian_kernel2d(self.kernel_size, self.sigma, prior.device, prior.dtype)
        prior = F.conv2d(prior, kernel, padding=self.kernel_size // 2)
        prior_min = prior.amin(dim=(-2, -1), keepdim=True)
        prior_max = prior.amax(dim=(-2, -1), keepdim=True)
        prior = (prior - prior_min) / (prior_max - prior_min + self.eps)
        return prior.squeeze(1)


def compute_step_semantic_prior(
    attn_probs: torch.Tensor,
    *,
    text_n: int,
    latent_n: int,
    height: int,
    width: int,
    kernel_size: int = 5,
    sigma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    block = extract_text_to_latent_block(attn_probs, text_n=text_n, latent_n=latent_n)
    spatial = block.mean(dim=(1, 2)).view(-1, 1, height, width)
    kernel = _gaussian_kernel2d(kernel_size, sigma, spatial.device, spatial.dtype)
    prior = F.conv2d(spatial, kernel, padding=kernel_size // 2)
    prior_min = prior.amin(dim=(-2, -1), keepdim=True)
    prior_max = prior.amax(dim=(-2, -1), keepdim=True)
    prior = (prior - prior_min) / (prior_max - prior_min + eps)
    return prior.squeeze(1)
