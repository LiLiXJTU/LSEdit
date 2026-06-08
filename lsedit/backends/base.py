from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class BackendRunContext:
    latents: Any = None
    reference_latents: Any = None
    latent_ids: Any = None
    reference_ids: Any = None
    prompt_embeds: Any = None
    text_ids: Any = None
    negative_prompt_embeds: Any = None
    prompt_embeds_mask: Any = None
    negative_prompt_embeds_mask: Any = None
    negative_text_ids: Any = None
    pooled_prompt_embeds: Any = None
    negative_pooled_prompt_embeds: Any = None
    guidance: Any = None
    img_shapes: Any = None
    # true_cfg_scale: float = 1.0
    do_true_cfg: bool = False
    height: int = 0
    width: int = 0
    image_height: int = 0
    image_width: int = 0
    text_n: int = 0
    latent_n: int = 0
    ref_n: int = 0


@runtime_checkable
class BackendAdapter(Protocol):
    backend_name: str

    def load_pipeline(self, runtime_cfg: Any) -> Any:
        ...

    def iter_transformer_blocks(self, pipeline: Any):
        ...

    def check_inputs(self, pipeline: Any, params: Any, callback_on_step_end_tensor_inputs: Any) -> None:
        ...

    def build_run_context(self, pipeline: Any, params: Any) -> BackendRunContext:
        ...

    def build_step_context(self, pipeline: Any, run_ctx: BackendRunContext, latents: Any, timestep: Any, guidance_scale: Any) -> Any:
        ...

    def run_cond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
        ...

    def run_uncond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
        ...

    def scheduler_step(self, pipeline: Any, noise_pred: Any, timestep: Any, latents: Any) -> Any:
        ...

    def decode_latents(self, pipeline: Any, latents: Any, run_ctx: BackendRunContext, output_type: Any, return_dict: Any) -> Any:
        ...
