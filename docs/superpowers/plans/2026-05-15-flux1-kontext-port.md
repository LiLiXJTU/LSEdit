# FLUX.1-Kontext Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port HAVEdit (WSP + HAVSR + BHC + Subject Release) from FLUX.2-klein-base-9B to FLUX.1-Kontext-dev so that the 2-backbone × 3-ablation matrix (baseline / w/o-BHC / full) on PIE-Bench can run end-to-end.

**Architecture:** Two new self-contained modules — an attention processor and a backend adapter — that share zero attention-processor code with the existing FLUX.2 path but reuse the four pure-function utility modules (`head_scores`, `semantic_prior`, `boundary_consistency`, `subject_mask_headwise`). Drive ablation via shell script wrapping existing `run_piebench_batch.py`.

**Tech Stack:** Python 3.10+, PyTorch 2.x, diffusers (`FluxKontextPipeline`, `FluxAttention`, `FluxAttnProcessor`, `FlowMatchEulerDiscreteScheduler`), pytest, bash.

**Reference docs:** Spec at `docs/superpowers/specs/2026-05-15-flux1-kontext-port-design.md`.

**Repo not under git:** `git add` / `git commit` steps are intentionally omitted from each task. If you initialize git later, batch-commit per task is recommended.

---

## Task 1: Scaffold the Backend Adapter and First Test

**Files:**
- Create: `havedit/backends/flux1_kontext.py`
- Create: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_flux1_kontext_adapter.py` with:

```python
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from havedit.backends import get_backend_adapter_class


def test_get_backend_adapter_class_returns_flux1_kontext():
    cls = get_backend_adapter_class("flux1-kontext")
    assert cls.backend_name == "flux1-kontext"
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /mnt/c/code/HAVEdit-main_changed
python -m pytest tests/test_flux1_kontext_adapter.py::test_get_backend_adapter_class_returns_flux1_kontext -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'havedit.backends.flux1_kontext'`.

- [ ] **Step 3: Write minimal adapter scaffold**

Create `havedit/backends/flux1_kontext.py`:

```python
from __future__ import annotations

from typing import Any

from havedit.backends.base import BackendRunContext


class Flux1KontextBackendAdapter:
    backend_name = "flux1-kontext"

    def load_pipeline(self, runtime_cfg: Any) -> Any:
        raise NotImplementedError

    def iter_transformer_blocks(self, pipeline: Any):
        raise NotImplementedError

    def check_inputs(self, pipeline: Any, params: Any, callback_on_step_end_tensor_inputs: Any) -> None:
        raise NotImplementedError

    def build_run_context(self, pipeline: Any, params: Any) -> BackendRunContext:
        raise NotImplementedError

    def build_step_context(self, pipeline: Any, run_ctx: BackendRunContext, latents: Any, timestep: Any, guidance_scale: Any) -> Any:
        raise NotImplementedError

    def run_cond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
        raise NotImplementedError

    def run_uncond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
        raise NotImplementedError

    def should_run_uncond_step(self, pipeline: Any, run_ctx: BackendRunContext, guidance_scale: Any) -> bool:
        return False

    def scheduler_step(self, pipeline: Any, noise_pred: Any, timestep: Any, latents: Any) -> Any:
        raise NotImplementedError

    def decode_latents(self, pipeline: Any, latents: Any, run_ctx: BackendRunContext, output_type: Any, return_dict: Any) -> Any:
        raise NotImplementedError

    def prepare_condition_images(self, pipeline: Any, image: Any, height: int | None, width: int | None):
        raise NotImplementedError

    def build_attention_processor(self, *, state: Any, block_name: str, stream_kind: str, default_processor_cls: Any) -> Any:
        raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_get_backend_adapter_class_returns_flux1_kontext -v
```

Expected: PASS.

---

## Task 2: Implement `iter_transformer_blocks` — Only Double-Stream

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock

from havedit.backends.flux1_kontext import Flux1KontextBackendAdapter


def test_iter_transformer_blocks_yields_only_double_stream():
    pipeline = SimpleNamespace(transformer=SimpleNamespace(
        transformer_blocks=[MagicMock(), MagicMock(), MagicMock()],
        single_transformer_blocks=[MagicMock(), MagicMock()],
    ))
    adapter = Flux1KontextBackendAdapter()
    yielded = list(adapter.iter_transformer_blocks(pipeline))
    assert len(yielded) == 3, f"expected 3 blocks, got {len(yielded)}"
    assert [name for _, name, _ in yielded] == ["ds_0", "ds_1", "ds_2"]
    assert all(stream == "double_stream" for _, _, stream in yielded)
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_iter_transformer_blocks_yields_only_double_stream -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the method**

Replace `iter_transformer_blocks` in `havedit/backends/flux1_kontext.py`:

```python
    def iter_transformer_blocks(self, pipeline: Any):
        transformer = getattr(pipeline, "transformer", pipeline)
        for index, block in enumerate(getattr(transformer, "transformer_blocks", [])):
            yield block, f"ds_{index}", "double_stream"
        # NOTE: single_transformer_blocks intentionally NOT yielded. FLUX.1-Kontext
        # single-stream blocks retain their default FluxAttnProcessor.
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_iter_transformer_blocks_yields_only_double_stream -v
```

Expected: PASS.

---

## Task 3: Implement `prepare_condition_images` — Snap to Kontext Resolutions

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
from PIL import Image

def test_prepare_condition_images_snaps_to_kontext_resolution():
    # Build a fake pipeline whose image_processor records the (height, width)
    # passed to resize/preprocess.
    recorded = {}

    def fake_resize(img, h, w):
        recorded["resize"] = (h, w)
        return img

    def fake_preprocess(img, h, w):
        recorded["preprocess"] = (h, w)
        return img

    pipeline = SimpleNamespace(
        vae_scale_factor=8,
        image_processor=SimpleNamespace(
            get_default_height_width=lambda img: (img.height, img.width),
            resize=fake_resize,
            preprocess=fake_preprocess,
        ),
    )

    src_image = Image.new("RGB", (800, 1200))  # aspect ratio 800/1200 ≈ 0.667
    adapter = Flux1KontextBackendAdapter()
    out_image, out_h, out_w = adapter.prepare_condition_images(pipeline, src_image, None, None)

    # PREFERRED_KONTEXT_RESOLUTIONS includes (672, 1568), (688, 1504), (720, 1456),
    # (752, 1392), (800, 1328), (832, 1248), (880, 1184) etc.
    # The closest to ratio 800/1200=0.667 is (832, 1248) ratio 832/1248 ≈ 0.667 — exact match.
    expected_h = 1248
    expected_w = 832
    multiple = pipeline.vae_scale_factor * 2  # 16
    expected_h = (expected_h // multiple) * multiple
    expected_w = (expected_w // multiple) * multiple
    assert recorded["resize"] == (expected_h, expected_w), recorded
    assert out_h == expected_h
    assert out_w == expected_w
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_prepare_condition_images_snaps_to_kontext_resolution -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the method**

Add `prepare_condition_images` to `havedit/backends/flux1_kontext.py`:

```python
    def prepare_condition_images(self, pipeline, image, height, width):
        if image is None:
            return None, height, width

        if not isinstance(image, list):
            image = [image]

        image_processor = getattr(pipeline, "image_processor", None)
        if image_processor is None:
            return image, height, width

        # Import lazily so tests can mock-import this module without diffusers installed.
        from diffusers.pipelines.flux.pipeline_flux_kontext import PREFERRED_KONTEXT_RESOLUTIONS

        single_image = image[0]
        if isinstance(single_image, type(None)) or (
            hasattr(single_image, "ndim") and single_image.ndim >= 4
            and single_image.shape[1] == getattr(pipeline, "latent_channels", -1)
        ):
            return image, height, width

        get_default = getattr(image_processor, "get_default_height_width", None)
        if not callable(get_default):
            return image, height, width

        image_height, image_width = get_default(single_image)
        if image_height is None or image_width is None or image_height <= 0:
            return image, height, width

        aspect_ratio = image_width / image_height
        _, image_width, image_height = min(
            (abs(aspect_ratio - w / h), w, h)
            for w, h in PREFERRED_KONTEXT_RESOLUTIONS
        )

        multiple_of = pipeline.vae_scale_factor * 2
        image_width = (image_width // multiple_of) * multiple_of
        image_height = (image_height // multiple_of) * multiple_of

        resize = getattr(image_processor, "resize", None)
        if callable(resize):
            image = resize(image, image_height, image_width)
        image = image_processor.preprocess(image, image_height, image_width)

        return image, image_height, image_width
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_prepare_condition_images_snaps_to_kontext_resolution -v
```

Expected: PASS.

---

## Task 4: Implement `build_run_context` — Capture Pooled Embeds

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
import torch


def test_build_run_context_captures_pooled_embeds():
    prompt_embeds = torch.zeros(1, 4, 16)        # text_n = 4
    pooled = torch.zeros(1, 16)
    text_ids = torch.zeros(4, 3)
    latents = torch.zeros(1, 256, 64)             # latent_n = 256
    image_latents = torch.zeros(1, 256, 64)       # ref_n = 256
    latent_ids = torch.zeros(256, 3)
    image_ids = torch.zeros(256, 3)
    image_ids[..., 0] = 1.0

    pipeline = MagicMock()
    pipeline._execution_device = "cpu"
    pipeline.do_classifier_free_guidance = False
    pipeline.encode_prompt = MagicMock(return_value=(prompt_embeds, pooled, text_ids))
    pipeline.prepare_latents = MagicMock(return_value=(latents, image_latents, latent_ids, image_ids))
    pipeline.transformer = SimpleNamespace(config=SimpleNamespace(in_channels=64))

    adapter = Flux1KontextBackendAdapter()
    params = SimpleNamespace(
        prompt="hello",
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        num_images_per_prompt=1,
        max_sequence_length=512,
        text_encoder_out_layers=None,
        generator=None,
        latents=None,
        height=512,
        width=512,
        condition_images=[Image.new("RGB", (512, 512))],
        guidance_scale=4.0,
        batch_size=1,
    )
    ctx = adapter.build_run_context(pipeline, params)
    assert ctx.text_n == 4
    assert ctx.latent_n == 256
    assert ctx.ref_n == 256
    assert ctx.pooled_prompt_embeds is pooled
    assert ctx.prompt_embeds is prompt_embeds
    assert ctx.text_ids is text_ids
    assert ctx.do_true_cfg is False
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_build_run_context_captures_pooled_embeds -v
```

Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement the method**

Add to `havedit/backends/flux1_kontext.py`:

```python
    def build_run_context(self, pipeline, params):
        device = getattr(pipeline, "_execution_device", None)

        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=params.prompt,
            prompt_2=None,
            prompt_embeds=params.prompt_embeds,
            pooled_prompt_embeds=params.pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=params.num_images_per_prompt,
            max_sequence_length=params.max_sequence_length,
            lora_scale=None,
        )

        negative_prompt_embeds = None
        negative_pooled = None
        negative_text_ids = None
        # CFG (true_cfg_scale > 1) is intentionally not supported in this port.

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
            image=params.condition_images[0] if params.condition_images else None,
            batch_size=params.batch_size * params.num_images_per_prompt,
            num_channels_latents=num_channels_latents,
            height=params.height,
            width=params.width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=params.generator,
            latents=params.latents,
        )

        latent_height, latent_width = self._spatial_shape_from_ids(latent_ids, latents.shape[1])

        return BackendRunContext(
            latents=latents,
            reference_latents=image_latents,
            latent_ids=latent_ids,
            reference_ids=image_ids,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            negative_text_ids=negative_text_ids,
            height=latent_height,
            width=latent_width,
            image_height=params.height,
            image_width=params.width,
            text_n=prompt_embeds.shape[1],
            latent_n=latents.shape[1],
            ref_n=0 if image_latents is None else image_latents.shape[1],
            do_true_cfg=False,
        )

    @staticmethod
    def _spatial_shape_from_ids(token_ids, token_count):
        if token_ids is None or getattr(token_ids, "numel", lambda: 0)() == 0 or token_ids.shape[-1] < 3:
            return 1, token_count
        height = int(token_ids[..., 1].max().item()) + 1
        width = int(token_ids[..., 2].max().item()) + 1
        return height, width
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_build_run_context_captures_pooled_embeds -v
```

Expected: PASS.

---

## Task 5: Implement Remaining Adapter Primitives

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`

(No tests in this task; these methods are mechanical thin wrappers around diffusers calls. They will be exercised by the smoke run in Task 10.)

- [ ] **Step 1: Implement `load_pipeline`**

Add to `havedit/backends/flux1_kontext.py`:

```python
    def load_pipeline(self, runtime_cfg):
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
        from havedit.flux.model_loader import configure_pipeline_runtime, resolve_dtype

        pipe = FluxKontextPipeline.from_pretrained(
            runtime_cfg.model_path,
            torch_dtype=resolve_dtype(runtime_cfg.torch_dtype),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        return configure_pipeline_runtime(pipe, runtime_cfg)
```

- [ ] **Step 2: Implement `check_inputs`**

```python
    def check_inputs(self, pipeline, params, callback_on_step_end_tensor_inputs):
        pipeline.check_inputs(
            prompt=params.prompt,
            prompt_2=None,
            height=params.height,
            width=params.width,
            negative_prompt=params.negative_prompt,
            negative_prompt_2=None,
            prompt_embeds=params.prompt_embeds,
            negative_prompt_embeds=params.negative_prompt_embeds,
            pooled_prompt_embeds=getattr(params, "pooled_prompt_embeds", None),
            negative_pooled_prompt_embeds=getattr(params, "negative_pooled_prompt_embeds", None),
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=params.max_sequence_length,
        )
```

- [ ] **Step 3: Implement `build_step_context`**

```python
    def build_step_context(self, pipeline, run_ctx, latents, timestep, guidance_scale):
        import torch
        latent_model_input = latents.to(pipeline.transformer.dtype)
        latent_image_ids = run_ctx.latent_ids
        if run_ctx.reference_latents is not None:
            latent_model_input = torch.cat(
                [latents, run_ctx.reference_latents], dim=1
            ).to(pipeline.transformer.dtype)
            # FluxKontext ids are 2D (seq, 3); concat along seq dim = dim 0.
            latent_image_ids = torch.cat(
                [run_ctx.latent_ids, run_ctx.reference_ids], dim=0
            )

        guidance = None
        if pipeline.transformer.config.guidance_embeds:
            guidance = torch.full(
                [1], guidance_scale, device=latents.device, dtype=torch.float32
            ).expand(latents.shape[0])

        return {
            "latent_model_input": latent_model_input,
            "latent_image_ids": latent_image_ids,
            "timestep": timestep.expand(latents.shape[0]).to(latents.dtype),
            "guidance": guidance,
        }
```

- [ ] **Step 4: Implement `run_cond_step`**

```python
    def run_cond_step(self, pipeline, state, step_ctx):
        run_ctx = state
        return pipeline.transformer(
            hidden_states=step_ctx["latent_model_input"],
            timestep=step_ctx["timestep"] / 1000,
            guidance=step_ctx["guidance"],
            pooled_projections=run_ctx.pooled_prompt_embeds,
            encoder_hidden_states=run_ctx.prompt_embeds,
            txt_ids=run_ctx.text_ids,
            img_ids=step_ctx["latent_image_ids"],
            joint_attention_kwargs=pipeline.attention_kwargs if hasattr(pipeline, "attention_kwargs") else None,
            return_dict=False,
        )[0]
```

- [ ] **Step 5: Implement `run_uncond_step` (only called when `should_run_uncond_step` returns True; not used in this port)**

```python
    def run_uncond_step(self, pipeline, state, step_ctx):
        run_ctx = state
        return pipeline.transformer(
            hidden_states=step_ctx["latent_model_input"],
            timestep=step_ctx["timestep"] / 1000,
            guidance=step_ctx["guidance"],
            pooled_projections=run_ctx.negative_pooled_prompt_embeds,
            encoder_hidden_states=run_ctx.negative_prompt_embeds,
            txt_ids=run_ctx.negative_text_ids,
            img_ids=step_ctx["latent_image_ids"],
            joint_attention_kwargs=pipeline.attention_kwargs if hasattr(pipeline, "attention_kwargs") else None,
            return_dict=False,
        )[0]
```

- [ ] **Step 6: Implement `scheduler_step`**

```python
    def scheduler_step(self, pipeline, noise_pred, timestep, latents):
        return pipeline.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
```

- [ ] **Step 7: Implement `decode_latents`**

```python
    def decode_latents(self, pipeline, latents, run_ctx, output_type, return_dict):
        if output_type == "latent":
            return latents

        vae = pipeline.vae
        image_processor = pipeline.image_processor

        latents = pipeline._unpack_latents(
            latents, run_ctx.image_height, run_ctx.image_width, pipeline.vae_scale_factor
        )
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
        image = vae.decode(latents, return_dict=False)[0]
        return image_processor.postprocess(image, output_type=output_type)
```

- [ ] **Step 8: Quick syntax/import sanity check**

```
python -c "from havedit.backends.flux1_kontext import Flux1KontextBackendAdapter; print(Flux1KontextBackendAdapter.backend_name)"
```

Expected: `flux1-kontext`

---

## Task 6: Scaffold Attention Processor + WSP Warmup Test

**Files:**
- Create: `havedit/flux/attention_processor_flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
from havedit.config import HAVEditConfig
from havedit.flux.runtime import HAVEditRuntimeState


def _make_state(text_n=4, latent_n=16, ref_n=16, height=4, width=4):
    state = HAVEditRuntimeState(config=HAVEditConfig())
    state.begin_run(total_steps=28, text_n=text_n, latent_n=latent_n, ref_n=ref_n, height=height, width=width)
    return state


def test_processor_accumulates_semantic_prior_in_warmup():
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    state = _make_state()
    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")

    batch, heads, total_seq = 1, 2, state.text_n + state.latent_n + state.ref_n
    attn_probs = torch.ones(batch, heads, state.text_n, total_seq) / total_seq

    for step in range(state.config.wsp.warmup_steps):
        state.begin_step(step)
        processor._maybe_update_wsp(attn_probs)

    assert state.semantic_accumulator.count == state.config.wsp.warmup_steps


def test_processor_does_not_accumulate_after_warmup():
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    state = _make_state()
    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")
    batch, heads, total_seq = 1, 2, state.text_n + state.latent_n + state.ref_n
    attn_probs = torch.ones(batch, heads, state.text_n, total_seq) / total_seq

    state.begin_step(state.config.wsp.warmup_steps)  # one past warmup
    processor._maybe_update_wsp(attn_probs)
    assert state.semantic_accumulator.count == 0
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_accumulates_semantic_prior_in_warmup -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create the processor with WSP-only implementation**

Create `havedit/flux/attention_processor_flux1_kontext.py`:

```python
"""Self-contained HAV attention processor for FLUX.1-Kontext-dev double-stream blocks.

Independent from the FLUX.2 processor in attention_processor_headwise_subjectbg_subjectrelease.py.
Reuses pure-function utility modules but no shared attention logic.

Reference: diffusers.models.transformers.transformer_flux.FluxAttnProcessor
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from havedit.flux.semantic_prior import WarmupSemanticPrior  # noqa: F401 (used via state)

dispatch_attention_fn = None
apply_rotary_emb = None
_DIFFUSERS_UNAVAILABLE = object()


def _load_dispatch_attention_fn():
    global dispatch_attention_fn
    if callable(dispatch_attention_fn):
        return dispatch_attention_fn
    if dispatch_attention_fn is _DIFFUSERS_UNAVAILABLE:
        return None
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn as fn
    except ImportError:
        dispatch_attention_fn = _DIFFUSERS_UNAVAILABLE
        return None
    dispatch_attention_fn = fn
    return fn


def _load_apply_rotary_emb():
    global apply_rotary_emb
    if callable(apply_rotary_emb):
        return apply_rotary_emb
    if apply_rotary_emb is _DIFFUSERS_UNAVAILABLE:
        return None
    try:
        from diffusers.models.embeddings import apply_rotary_emb as fn
    except ImportError:
        apply_rotary_emb = _DIFFUSERS_UNAVAILABLE
        return None
    apply_rotary_emb = fn
    return fn


class FluxKontextHAVAttnProcessor:
    """HAV processor for FLUX.1-Kontext double-stream blocks.

    Hooks:
    - Warmup steps: accumulate text→image cross-attention into WSP.
    - Post-warmup: HAVSR replaces v_pred toward v_ref weighted by per-head preserve scores.
    - Optional BHC refinement (config.bhc.enabled).
    - Subject mask discovery and late-stage release.
    """

    def __init__(self, state, block_name: str):
        self.state = state
        self.block_name = block_name

    def _maybe_update_wsp(self, attn_probs: torch.Tensor) -> None:
        """attn_probs shape: [B, H, text_n, total_seq_len]."""
        if self.state.current_step >= self.state.config.wsp.warmup_steps:
            return
        if self.state.text_n <= 0 or self.state.latent_n <= 0:
            return
        self.state.semantic_accumulator.update(
            attn_probs,
            text_n=self.state.text_n,
            latent_n=self.state.latent_n,
        )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ):
        # Standard FluxAttn QKV extraction.
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if encoder_hidden_states is not None and getattr(attn, "added_kv_proj_dim", None) is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_key = attn.add_k_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_value = attn.add_v_proj(encoder_hidden_states).unflatten(-1, (attn.heads, -1))
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            rotary = _load_apply_rotary_emb()
            if rotary is not None:
                query = rotary(query, image_rotary_emb, sequence_dim=1)
                key = rotary(key, image_rotary_emb, sequence_dim=1)

        # Always materialize text-query × all-keys attention (cheap; only text rows).
        # Used by both WSP accumulation (warmup only) and latest_semantic_prior_current
        # (every step; needed for Task 9 subject mask freezing).
        if (
            attention_mask is None
            and self.state.text_n > 0 and self.state.latent_n > 0
        ):
            q_text = query.transpose(1, 2)[:, :, : self.state.text_n, :].float()
            k_all = key.transpose(1, 2).float()
            scale = q_text.shape[-1] ** -0.5
            attn_scores = torch.matmul(q_text, k_all.transpose(-1, -2)) * scale
            attn_probs = attn_scores.softmax(dim=-1).to(query.dtype)
            if hasattr(self.state, "update_semantic_prior_current"):
                self.state.update_semantic_prior_current(
                    attn_probs=attn_probs,
                    text_n=self.state.text_n,
                    latent_n=self.state.latent_n,
                )
            self._maybe_update_wsp(attn_probs)  # has its own warmup gate

        # Dispatch attention with (possibly modified) value.
        dispatch = _load_dispatch_attention_fn()
        if dispatch is not None:
            hidden_states = dispatch(query, key, value, attn_mask=attention_mask, backend=None)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            ).transpose(1, 2)

        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_accumulates_semantic_prior_in_warmup tests/test_flux1_kontext_adapter.py::test_processor_does_not_accumulate_after_warmup -v
```

Expected: PASS (both tests).

---

## Task 7: Implement HAVSR — Replace Pred Values Only

**Files:**
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
def test_processor_havsr_replaces_only_pred_values():
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    torch.manual_seed(0)
    state = _make_state(text_n=4, latent_n=16, ref_n=16, height=4, width=4)
    state.begin_step(state.config.wsp.warmup_steps)  # past warmup
    state.semantic_prior = torch.zeros(1, state.height, state.width)  # no semantic prior signal

    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")

    # value shape: [B, total_seq, H, D]
    batch, heads, head_dim = 1, 2, 8
    total = state.text_n + state.latent_n + state.ref_n
    value = torch.randn(batch, total, heads, head_dim)
    value_before = value.clone()

    pred_start = state.text_n
    pred_end = pred_start + state.latent_n
    ref_start = pred_end
    ref_end = ref_start + state.ref_n

    modified = processor._maybe_replace_pred_values(value)

    # ref slice must be byte-for-byte identical
    assert torch.equal(modified[:, ref_start:ref_end, :, :], value_before[:, ref_start:ref_end, :, :])
    # text slice must be unchanged
    assert torch.equal(modified[:, :pred_start, :, :], value_before[:, :pred_start, :, :])
    # pred slice must be at least different in some position (replacement happened)
    assert not torch.equal(modified[:, pred_start:pred_end, :, :], value_before[:, pred_start:pred_end, :, :])
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_havsr_replaces_only_pred_values -v
```

Expected: FAIL with `AttributeError: ... has no attribute '_maybe_replace_pred_values'`.

- [ ] **Step 3: Add HAVSR helpers and integrate into `__call__`**

Add to `havedit/flux/attention_processor_flux1_kontext.py`:

```python
from havedit.flux.head_scores import compute_head_deviation, _local_zscore_map
```

Then add methods to `FluxKontextHAVAttnProcessor`:

```python
    def _split_pred_ref(self, value_btransposed: torch.Tensor):
        """value_btransposed shape: [B, H, total_seq, D]. Return (v_pred, v_ref) slices."""
        pred_start = self.state.text_n
        pred_end = pred_start + self.state.latent_n
        ref_start = pred_end
        ref_end = ref_start + self.state.ref_n
        v_pred = value_btransposed[:, :, pred_start:pred_end, :]
        v_ref = value_btransposed[:, :, ref_start:ref_end, :]
        return v_pred, v_ref, pred_start, pred_end

    def _compute_preserve_weight(self, v_pred: torch.Tensor, v_ref: torch.Tensor):
        cfg = self.state.config.havsr
        semantic_prior = getattr(self.state, "latest_semantic_prior_current", None)
        if semantic_prior is None:
            semantic_prior = getattr(self.state, "semantic_prior", None)
        if semantic_prior is None:
            return None

        deviation = compute_head_deviation(v_pred, v_ref)
        z_dev = _local_zscore_map(
            deviation,
            height=self.state.height,
            width=self.state.width,
            kernel_size=cfg.local_kernel_size,
            eps=cfg.eps,
        )
        sem = semantic_prior.reshape(semantic_prior.shape[0], 1, -1).to(dtype=z_dev.dtype, device=z_dev.device)
        score = torch.sigmoid(-float(cfg.alpha) * z_dev + float(cfg.beta) * (1.0 - sem))
        soft_band = max((1.0 - float(cfg.threshold)) * 0.5, float(cfg.eps))
        full_at = float(cfg.threshold) + soft_band
        weight = ((score - float(cfg.threshold)) / soft_band).clamp(0.0, 1.0)
        weight = torch.where(score >= full_at, torch.ones_like(weight), weight)
        return weight, score

    def _maybe_replace_pred_values(self, value: torch.Tensor) -> torch.Tensor:
        """value shape: [B, total_seq, H, D]. Returns same shape with pred slice possibly replaced."""
        if self.state.current_step < self.state.config.wsp.warmup_steps:
            return value
        if self.state.text_n <= 0 or self.state.latent_n <= 0 or self.state.ref_n <= 0:
            return value
        if value.shape[1] < self.state.text_n + self.state.latent_n + self.state.ref_n:
            return value

        value_bh = value.transpose(1, 2)  # [B, H, total_seq, D]
        v_pred, v_ref, pred_start, pred_end = self._split_pred_ref(value_bh)
        if v_pred.shape[2] == 0 or v_pred.shape[2] != v_ref.shape[2]:
            return value

        weight_and_score = self._compute_preserve_weight(v_pred, v_ref)
        if weight_and_score is None:
            return value
        preserve_weight, _preserve_score = weight_and_score

        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)
        v_pred_new = v_ref + gain * (v_pred - v_ref)

        value_bh = value_bh.clone()
        value_bh[:, :, pred_start:pred_end, :] = v_pred_new
        return value_bh.transpose(1, 2)
```

Then modify `__call__` to invoke it. Replace this block in `__call__`:

```python
        # Dispatch attention with (possibly modified) value.
```

with:

```python
        # HAVSR: replace pred values before dispatching attention.
        value = self._maybe_replace_pred_values(value)

        # Dispatch attention with (possibly modified) value.
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_havsr_replaces_only_pred_values -v
```

Expected: PASS.

---

## Task 8: BHC Toggle and Application

**Files:**
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`
- Modify: `tests/test_flux1_kontext_adapter.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_flux1_kontext_adapter.py`:

```python
from unittest.mock import patch


def test_processor_bhc_disabled_skips_application():
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    torch.manual_seed(0)
    state = _make_state()
    state.begin_step(state.config.wsp.warmup_steps)
    state.semantic_prior = torch.zeros(1, state.height, state.width)
    state.config.bhc.enabled = False

    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")
    batch, heads, head_dim = 1, 2, 8
    total = state.text_n + state.latent_n + state.ref_n
    value = torch.randn(batch, total, heads, head_dim)

    with patch(
        "havedit.flux.attention_processor_flux1_kontext.apply_boundary_head_consistency"
    ) as mocked_bhc:
        processor._maybe_replace_pred_values(value)
        assert mocked_bhc.call_count == 0


def test_processor_bhc_enabled_invokes_application():
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    torch.manual_seed(0)
    state = _make_state()
    state.begin_step(state.config.wsp.warmup_steps)
    state.semantic_prior = torch.zeros(1, state.height, state.width)
    state.config.bhc.enabled = True

    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")
    batch, heads, head_dim = 1, 2, 8
    total = state.text_n + state.latent_n + state.ref_n
    value = torch.randn(batch, total, heads, head_dim)

    # Patch BHC to return its `values` input unchanged so we can assert it was called.
    with patch(
        "havedit.flux.attention_processor_flux1_kontext.apply_boundary_head_consistency",
        side_effect=lambda values, refs, scores, tau_low, tau_high, lambda_max: values,
    ) as mocked_bhc:
        processor._maybe_replace_pred_values(value)
        assert mocked_bhc.call_count == 1
```

- [ ] **Step 2: Run to verify FAIL**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_bhc_disabled_skips_application tests/test_flux1_kontext_adapter.py::test_processor_bhc_enabled_invokes_application -v
```

Expected: FAIL (both — `apply_boundary_head_consistency` not imported / not invoked).

- [ ] **Step 3: Add BHC integration**

Add import to `havedit/flux/attention_processor_flux1_kontext.py` (after the existing `head_scores` import):

```python
from havedit.flux.boundary_consistency import apply_boundary_head_consistency
```

Modify `_maybe_replace_pred_values` to apply BHC after the soft replacement. Replace:

```python
        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)
        v_pred_new = v_ref + gain * (v_pred - v_ref)

        value_bh = value_bh.clone()
        value_bh[:, :, pred_start:pred_end, :] = v_pred_new
        return value_bh.transpose(1, 2)
```

with:

```python
        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)
        v_pred_new = v_ref + gain * (v_pred - v_ref)

        if self.state.config.bhc.enabled:
            v_pred_new = apply_boundary_head_consistency(
                values=v_pred_new,
                refs=v_ref,
                scores=_preserve_score,
                tau_low=self.state.config.bhc.tau_low,
                tau_high=self.state.config.bhc.tau_high,
                lambda_max=self.state.config.bhc.lambda_max,
            )

        value_bh = value_bh.clone()
        value_bh[:, :, pred_start:pred_end, :] = v_pred_new
        return value_bh.transpose(1, 2)
```

- [ ] **Step 4: Run to verify PASS**

```
python -m pytest tests/test_flux1_kontext_adapter.py::test_processor_bhc_disabled_skips_application tests/test_flux1_kontext_adapter.py::test_processor_bhc_enabled_invokes_application -v
```

Expected: PASS (both).

---

## Task 9: Wire Adapter's `build_attention_processor` + Subject Release + Clean Up Old Branches

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`
- Modify: `havedit/flux/attention_processor_headwise_subjectbg_subjectrelease.py`
- Modify: `havedit/flux/warp.py`

- [ ] **Step 1: Add subject release path to the processor**

In `havedit/flux/attention_processor_flux1_kontext.py`, add the imports:

```python
from havedit.flux.subject_mask_headwise import (
    build_subject_candidate, refine_subject_candidate,
    select_subject_region, dilate_subject_region,
)
```

Add a helper method to `FluxKontextHAVAttnProcessor`:

```python
    def _is_last_double_stream_block(self) -> bool:
        if not self.block_name.startswith("ds_"):
            return False
        block_count = int(getattr(self.state, "double_stream_block_count", 0))
        if block_count <= 0:
            return self.block_name == "ds_0"
        return self.block_name == f"ds_{block_count - 1}"

    def _maybe_freeze_subject_mask(self) -> None:
        if (self.state.current_step + 1) != int(getattr(self.state, "background_discovery_step", 16)):
            return
        if not self._is_last_double_stream_block():
            return
        if getattr(self.state, "frozen_background_mask", None) is not None:
            return
        semantic_prior_current = getattr(self.state, "latest_semantic_prior_current", None)
        if semantic_prior_current is None:
            return
        candidate = build_subject_candidate(
            semantic_prior_current,
            subject_threshold=float(getattr(self.state, "subject_threshold", 0.4)),
        )
        candidate = refine_subject_candidate(
            candidate,
            open_kernel=int(getattr(self.state, "subject_open_kernel", 1)),
            close_kernel=int(getattr(self.state, "subject_close_kernel", 1)),
        )
        core = semantic_prior_current >= float(self.state.config.havsr.threshold)
        subject = select_subject_region(
            candidate,
            mode=str(getattr(self.state, "subject_select_mode", "largest")),
            core=core,
        )
        subject = dilate_subject_region(
            subject,
            radius=int(getattr(self.state, "subject_dilate_radius", 1)),
        )
        background = ~subject
        self.state.frozen_subject_mask = subject.detach().to(dtype=torch.float32).cpu()
        self.state.frozen_background_mask = background.detach().to(dtype=torch.float32).cpu()
```

In `_compute_preserve_weight`, immediately before `return weight, score`, add:

```python
        # Subject release in late steps.
        release_step = int(getattr(self.state, "subject_release_step", self.state.background_discovery_step))
        release_scale = float(getattr(self.state, "subject_release_scale", 1.0))
        frozen_subject = getattr(self.state, "frozen_subject_mask", None)
        if (self.state.current_step + 1) >= release_step and frozen_subject is not None and release_scale < 1.0:
            subject_mask = frozen_subject.to(dtype=weight.dtype, device=weight.device).reshape(
                frozen_subject.shape[0], 1, -1
            )
            weight = torch.where(subject_mask > 0, weight * release_scale, weight)
```

In `_maybe_replace_pred_values`, just before computing `weight_and_score`, call:

```python
        self._maybe_freeze_subject_mask()
```

- [ ] **Step 2: Implement `build_attention_processor` on the adapter**

Add to `havedit/backends/flux1_kontext.py`:

```python
    def build_attention_processor(self, *, state, block_name, stream_kind, default_processor_cls):
        from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor
        # FLUX.1-Kontext only attaches HAV to double-stream blocks. iter_transformer_blocks
        # never yields single-stream blocks, so stream_kind is always "double_stream" here.
        return FluxKontextHAVAttnProcessor(state=state, block_name=block_name)
```

- [ ] **Step 3: Delete old flux1-kontext branches**

In `havedit/flux/attention_processor_headwise_subjectbg_subjectrelease.py` around line 581, find:

```python
    for block, block_name, stream_kind in adapter.iter_transformer_blocks(pipeline):
        if backend_name == "flux1-kontext" and stream_kind == "single_stream":
            continue
        build_attention_processor = getattr(adapter, "build_attention_processor", None)
```

Replace with (delete the `if` block entirely):

```python
    for block, block_name, stream_kind in adapter.iter_transformer_blocks(pipeline):
        build_attention_processor = getattr(adapter, "build_attention_processor", None)
```

In `havedit/flux/warp.py` around line 451, find the analogous block in `enable_havedit`:

```python
    for block, block_name, stream_kind in adapter.iter_transformer_blocks(pipeline):
        attention = block.attn
        original_processors.append((attention, getattr(attention, "processor", None)))
        if backend_name == "flux1-kontext" and stream_kind == "single_stream":
            continue
        build_attention_processor = getattr(adapter, "build_attention_processor", None)
```

Replace with (delete the `if` block entirely):

```python
    for block, block_name, stream_kind in adapter.iter_transformer_blocks(pipeline):
        attention = block.attn
        original_processors.append((attention, getattr(attention, "processor", None)))
        build_attention_processor = getattr(adapter, "build_attention_processor", None)
```

In `havedit/flux/warp.py` around lines 79-108 inside `_prepare_condition_images`, find the entire `if backend_name == "flux1-kontext":` block (it spans the `if` line through the `return image, height, width` inside it). Delete the entire block. The function should fall through to the generic `condition_images = []` loop below.

For reference, the function should look like this after the edit:

```python
def _prepare_condition_images(pipeline, image, height, width):
    if image is None:
        return None, height, width

    backend_name = getattr(getattr(pipeline, "transformer", None), "_havedit_backend", None)  # kept; harmless
    if not isinstance(image, list):
        image = [image]

    image_processor = getattr(pipeline, "image_processor", None)
    if image_processor is None:
        return image, height, width

    condition_images = []
    for img in image:
        check_image_input = getattr(image_processor, "check_image_input", None)
        if callable(check_image_input):
            check_image_input(img)
        # ...rest unchanged...
```

- [ ] **Step 4: Run full adapter test suite to verify nothing regressed**

```
python -m pytest tests/test_flux1_kontext_adapter.py -v
```

Expected: ALL PASS (8+ tests).

- [ ] **Step 5: Run pre-existing FLUX.2 tests to verify no regression**

```
python -m pytest tests/test_existing_backends_unchanged.py tests/test_backend_warp_integration.py tests/test_flux2_adapter.py -v
```

Expected: ALL PASS.

---

## Task 10: Ablation Shell Script + Smoke Run

**Files:**
- Create: `scripts/run_dual_backbone_ablation.sh`

- [ ] **Step 1: Create the shell script**

Create `scripts/run_dual_backbone_ablation.sh` with mode `0755`:

```bash
#!/usr/bin/env bash
# Run 2-backbone × 3-ablation PIE-Bench matrix:
#   {FLUX.2-klein, FLUX.1-Kontext-dev} × {baseline, HAVEdit w/o BHC, HAVEdit full}
#
# Usage:
#   bash scripts/run_dual_backbone_ablation.sh                      # full PIE-Bench
#   bash scripts/run_dual_backbone_ablation.sh --limit 3            # smoke: 3 samples per cell
#   bash scripts/run_dual_backbone_ablation.sh --sample-ids 000000000005,924000000002
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FLUX2_PATH="${FLUX2_PATH:-/data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B}"
FLUX1K_PATH="${FLUX1K_PATH:-/data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev}"
PIE_ROOT="${PIE_ROOT:-/data/ll/data/PIEbench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/ablation_dual_backbone}"

EXTRA=("$@")

run_one() {
    local label="$1" backend="$2" model_path="$3"
    shift 3
    echo "==> [${label}] backend=${backend} model=${model_path}"
    python scripts/run_piebench_batch.py \
        --backend "${backend}" \
        --model-path "${model_path}" \
        --pie-root "${PIE_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/${label}" \
        --disable-trajectory-trust \
        "${EXTRA[@]}" \
        "$@"
}

run_one "flux2_baseline"        flux2          "${FLUX2_PATH}"  --disable-havedit
run_one "flux2_havedit_no_bhc"  flux2          "${FLUX2_PATH}"  --disable-bhc
run_one "flux2_havedit_full"    flux2          "${FLUX2_PATH}"
run_one "flux1k_baseline"       flux1-kontext  "${FLUX1K_PATH}" --disable-havedit
run_one "flux1k_havedit_no_bhc" flux1-kontext  "${FLUX1K_PATH}" --disable-bhc
run_one "flux1k_havedit_full"   flux1-kontext  "${FLUX1K_PATH}"

echo "==> All 6 cells finished. Output: ${OUTPUT_ROOT}"
```

Set executable:

```
chmod +x /mnt/c/code/HAVEdit-main_changed/scripts/run_dual_backbone_ablation.sh
```

- [ ] **Step 2: Run a 1-sample smoke on one FLUX.1-Kontext cell first**

(This is the actual E2E verification; needs GPU + weights.)

```
cd /mnt/c/code/HAVEdit-main_changed
python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev \
    --pie-root /data/ll/data/PIEbench \
    --output-dir output/smoke_flux1k_full \
    --disable-trajectory-trust \
    --limit 1
```

Expected:
- Pipeline loads without error
- 1 sample processed
- `output/smoke_flux1k_full/havedit_flux1_kontext_t28_g4_0_seedNone/<sample_id>/strength_1_00.jpg` exists
- No tracebacks in stdout

If this fails, debug before moving to Step 3.

- [ ] **Step 3: Run 3-sample smoke across all 6 cells**

```
bash scripts/run_dual_backbone_ablation.sh --limit 3
```

Expected:
- 6 print headers `==> [<label>] backend=... model=...`
- 6 corresponding output dirs created under `output/ablation_dual_backbone/`
- Each cell has 3 sample subdirs
- Final line: `==> All 6 cells finished. Output: ...`

- [ ] **Step 4: Spot-check outputs**

```
ls output/ablation_dual_backbone/*/havedit_*/
```

Expected: 6 subdirectories, each containing 3 sample folders with `strength_1_00.jpg` and `metadata.json`.

- [ ] **Step 5: Run full PIE-Bench (optional, overnight job)**

```
nohup bash scripts/run_dual_backbone_ablation.sh > output/ablation_full.log 2>&1 &
echo "Started PID $!"
```

Monitor via `tail -f output/ablation_full.log`.

---

## Spec Coverage Check

| Spec section | Implementing task |
| --- | --- |
| §3 Added: `havedit/flux/attention_processor_flux1_kontext.py` | Tasks 6, 7, 8, 9 |
| §3 Added: `havedit/backends/flux1_kontext.py` | Tasks 1, 2, 3, 4, 5, 9 |
| §3 Added: `scripts/run_dual_backbone_ablation.sh` | Task 10 |
| §3 Added: `tests/test_flux1_kontext_adapter.py` | Tasks 1-8 |
| §3 Deleted: attention_processor_headwise_subjectbg_subjectrelease.py:581 | Task 9 Step 3 |
| §3 Deleted: warp.py:451 | Task 9 Step 3 |
| §3 Deleted: warp.py:79-108 | Task 9 Step 3 |
| §4 New attention processor full flow | Tasks 6 (WSP), 7 (HAVSR), 8 (BHC), 9 (Subject Release) |
| §5 Backend adapter all methods | Tasks 1 (scaffold), 2 (iter), 3 (prepare_condition_images), 4 (build_run_context), 5 (rest), 9 (build_attention_processor) |
| §6 Ablation orchestration | Task 10 |
| §7 Testing | Tasks 1-8 unit tests + Task 10 smoke |
