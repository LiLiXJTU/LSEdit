from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Optional

import torch

from ..config import HAVEditConfig
from .semantic_prior import WarmupSemanticPrior


@dataclass
class HAVEditRuntimeState:
    config: HAVEditConfig
    current_step: int = 0
    total_steps: int = 0
    text_n: int = 0
    latent_n: int = 0
    ref_n: int = 0
    height: int = 0
    width: int = 0
    source_latents: Optional[torch.Tensor] = None
    latent_ids: Optional[torch.Tensor] = None
    semantic_prior: Optional[torch.Tensor] = None
    visualize_attention: bool = False
    latest_attention_map: Optional[torch.Tensor] = None
    attention_map_sum: Optional[torch.Tensor] = None
    attention_map_count: int = 0
    previous_delta: Optional[torch.Tensor] = None
    ema_delta: Optional[torch.Tensor] = None
    semantic_accumulator: WarmupSemanticPrior = field(init=False)
    latest_preserve_score_mean: Optional[float] = None
    latest_preserve_weight_mean: Optional[float] = None
    latest_consistency_score_mean: Optional[float] = None
    latest_release_score_mean: Optional[float] = None
    latest_trust_score_mean: Optional[float] = None
    latest_pred_velocity: Optional[torch.Tensor] = None
    latest_attention_js_ref_vs_pred: Optional[float] = None
    latest_edit_mask: Optional[torch.Tensor] = None
    edit_mask_sum: Optional[torch.Tensor] = None
    edit_mask_count: int = 0
    latest_preserve_weight_map: Optional[torch.Tensor] = None
    preserve_weight_sum: Optional[torch.Tensor] = None
    preserve_weight_count: int = 0
    latest_semantic_prior_current: Optional[torch.Tensor] = None
    latest_headwise_preserve_score: Optional[torch.Tensor] = None
    latest_headwise_preserve_weight: Optional[torch.Tensor] = None
    latest_headwise_zdev: Optional[torch.Tensor] = None
    latest_subject_candidate: Optional[torch.Tensor] = None
    latest_subject_mask: Optional[torch.Tensor] = None
    latest_background_mask: Optional[torch.Tensor] = None
    frozen_subject_mask: Optional[torch.Tensor] = None
    frozen_background_mask: Optional[torch.Tensor] = None
    double_stream_block_count: int = 0
    subject_threshold: float = 0.4
    subject_select_mode: str = "largest"
    subject_open_kernel: int = 1
    subject_close_kernel: int = 1
    subject_dilate_radius: int = 1
    background_discovery_step: int = 16
    subject_release_step: int = 16
    subject_release_scale: float = 1.0

    def __post_init__(self) -> None:
        self.semantic_accumulator = self._build_semantic_accumulator()

    def _build_semantic_accumulator(self) -> WarmupSemanticPrior:
        return WarmupSemanticPrior(
            kernel_size=self.config.wsp.gaussian_kernel_size,
            sigma=self.config.wsp.gaussian_sigma,
        )

    def begin_run(
        self,
        total_steps: int,
        text_n: int,
        latent_n: int,
        ref_n: int,
        height: int,
        width: int,
    ) -> None:
        self.total_steps = total_steps
        self.text_n = text_n
        self.latent_n = latent_n
        self.ref_n = ref_n
        self.height = height
        self.width = width
        self.current_step = 0
        self.source_latents = None
        self.latent_ids = None
        self.semantic_prior = None
        self.latest_attention_map = None
        self.attention_map_sum = None
        self.attention_map_count = 0
        self.previous_delta = None
        self.ema_delta = None
        self.semantic_accumulator = self._build_semantic_accumulator()
        self.latest_preserve_score_mean = None
        self.latest_preserve_weight_mean = None
        self.latest_consistency_score_mean = None
        self.latest_release_score_mean = None
        self.latest_trust_score_mean = None
        self.latest_pred_velocity = None
        self.latest_attention_js_ref_vs_pred = None
        self.latest_edit_mask = None
        self.edit_mask_sum = None
        self.edit_mask_count = 0
        self.latest_preserve_weight_map = None
        self.preserve_weight_sum = None
        self.preserve_weight_count = 0
        self.latest_semantic_prior_current = None
        self.latest_headwise_preserve_score = None
        self.latest_headwise_preserve_weight = None
        self.latest_headwise_zdev = None
        self.latest_subject_candidate = None
        self.latest_subject_mask = None
        self.latest_background_mask = None
        self.frozen_subject_mask = None
        self.frozen_background_mask = None

    def begin_step(self, step: int) -> None:
        self.current_step = step
        self.latest_edit_mask = None
        self.edit_mask_sum = None
        self.edit_mask_count = 0
        self.latest_preserve_weight_map = None
        self.preserve_weight_sum = None
        self.preserve_weight_count = 0
        self.latest_headwise_preserve_score = None
        self.latest_headwise_preserve_weight = None
        self.latest_headwise_zdev = None
        self.latest_subject_candidate = None
        self.latest_subject_mask = None
        self.latest_background_mask = None
        if self.visualize_attention:
            self.latest_attention_map = None
            self.attention_map_sum = None
            self.attention_map_count = 0

    def configure_visualization(self, *, visualize_attention: bool) -> None:
        self.visualize_attention = bool(visualize_attention)
        self.latest_attention_map = None
        self.attention_map_sum = None
        self.attention_map_count = 0

    def accumulate_attention_map(self, attention_map: torch.Tensor) -> None:
        if not self.visualize_attention:
            return

        if attention_map.ndim == 3:
            map_2d = attention_map[0]
        elif attention_map.ndim == 2:
            map_2d = attention_map
        else:
            return

        map_2d = map_2d.detach().to(dtype=torch.float32).cpu()
        if self.attention_map_sum is None:
            self.attention_map_sum = map_2d
            self.attention_map_count = 1
        else:
            self.attention_map_sum = self.attention_map_sum + map_2d
            self.attention_map_count += 1

        if self.attention_map_sum is not None and self.attention_map_count > 0:
            self.latest_attention_map = self.attention_map_sum / float(self.attention_map_count)

    def accumulate_edit_mask(self, edit_mask: torch.Tensor) -> None:
        if edit_mask.ndim != 3:
            return
        mask = edit_mask.detach().to(dtype=torch.float32).cpu()
        if self.edit_mask_sum is None:
            self.edit_mask_sum = mask
            self.edit_mask_count = 1
        else:
            if self.edit_mask_sum.shape != mask.shape:
                return
            self.edit_mask_sum = self.edit_mask_sum + mask
            self.edit_mask_count += 1
        if self.edit_mask_sum is not None and self.edit_mask_count > 0:
            self.latest_edit_mask = self.edit_mask_sum / float(self.edit_mask_count)

    def accumulate_preserve_weight_map(self, preserve_weight_map: torch.Tensor) -> None:
        if preserve_weight_map.ndim != 3:
            return
        weight_map = preserve_weight_map.detach().to(dtype=torch.float32).cpu()
        if self.preserve_weight_sum is None:
            self.preserve_weight_sum = weight_map
            self.preserve_weight_count = 1
        else:
            if self.preserve_weight_sum.shape != weight_map.shape:
                return
            self.preserve_weight_sum = self.preserve_weight_sum + weight_map
            self.preserve_weight_count += 1
        if self.preserve_weight_sum is not None and self.preserve_weight_count > 0:
            self.latest_preserve_weight_map = self.preserve_weight_sum / float(self.preserve_weight_count)

    def finalize_semantic_prior_if_needed(self) -> None:
        if self.semantic_prior is None and self.current_step == self.config.wsp.warmup_steps - 1:
            self.semantic_prior = self.semantic_accumulator.finalize(self.height, self.width)

    def update_semantic_prior_current(
        self,
        *,
        attn_probs: torch.Tensor,
        text_n: int,
        latent_n: int,
    ) -> None:
        if self.height <= 0 or self.width <= 0:
            return
        from .semantic_prior import compute_step_semantic_prior

        self.latest_semantic_prior_current = compute_step_semantic_prior(
            attn_probs,
            text_n=text_n,
            latent_n=latent_n,
            height=self.height,
            width=self.width,
            kernel_size=self.config.wsp.gaussian_kernel_size,
            sigma=self.config.wsp.gaussian_sigma,
        )

    def begin_reference_value_capture(self) -> None:
        self.capture_reference_values = True
        self.cached_reference_values_by_block = {}

    def end_reference_value_capture(self) -> None:
        self.capture_reference_values = False

    def cache_reference_values(self, block_name: str, values: torch.Tensor) -> None:
        self.cached_reference_values_by_block[block_name] = values.detach()

    def get_cached_reference_values(self, block_name: str) -> Optional[torch.Tensor]:
        return self.cached_reference_values_by_block.get(block_name)
