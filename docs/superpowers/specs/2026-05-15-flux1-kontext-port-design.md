# FLUX.1-Kontext-dev Port for HAVEdit Comparison Experiments — Design

**Date:** 2026-05-15
**Author:** L.L.
**Status:** Draft, awaiting user review

## 1. Goal

Validate that HAVEdit's three remaining innovations — **WSP** (Warmup Semantic Prior),
**HAVSR** (Head-Aware Value Selective Replacement), **BHC** (Boundary Head Consistency),
and **Subject-Background detection + Subject Release** — transfer from FLUX.2-klein-base-9B
to FLUX.1-Kontext-dev.

`trajectory_trust` is explicitly excluded: prior author investigation showed it did
not improve results, so it is disabled in every cell of this experiment.

The output is a 2-backbone × 3-ablation matrix on PIE-Bench, suitable for the
paper's comparison table:

|                       | FLUX.2-klein | FLUX.1-Kontext-dev |
| --------------------- | :----------: | :----------------: |
| baseline (no HAVEdit) |       ✓      |          ✓         |
| HAVEdit w/o BHC       |       ✓      |          ✓         |
| HAVEdit full          |       ✓      |          ✓         |

## 2. Constraints and Assumptions

- FLUX.1-Kontext-dev weights live at `/data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev`,
  parallel to the existing FLUX.2 path.
- PIE-Bench data and `run_piebench_batch.py` are already in place and support
  `--backend flux1-kontext`, `--disable-bhc`, `--disable-havedit`,
  `--disable-trajectory-trust` flags.
- The four shared utility modules (`head_scores`, `semantic_prior`,
  `boundary_consistency`, `subject_mask_headwise`) are pure functions and
  unchanged.
- `HAVEditRuntimeState` (`havedit/flux/runtime.py`) is reused as-is.
- `CFG` (true_cfg_scale > 1) is **not supported** in this port. We always run
  with the CFG-distilled guidance pathway. Adding negative-prompt CFG later is
  a follow-up.

## 3. File Layout

### Added

| Path | Purpose |
| --- | --- |
| `havedit/flux/attention_processor_flux1_kontext.py` | New, self-contained HAV processor for FLUX.1-Kontext double-stream blocks. |
| `havedit/backends/flux1_kontext.py` | New `Flux1KontextBackendAdapter` implementing the `BackendAdapter` protocol. |
| `scripts/run_dual_backbone_ablation.sh` | Shell orchestrator that runs `run_piebench_batch.py` six times. |
| `tests/test_flux1_kontext_adapter.py` | Mock-based unit tests for the adapter and processor. |

### Deleted / Cleaned

These existing branches were written speculatively for `flux1-kontext` and are
suspected to be incorrect. Removing them reflects the strict-separation design:

| File | Lines (approx) | Action |
| --- | --- | --- |
| `havedit/flux/attention_processor_headwise_subjectbg_subjectrelease.py` | 581 | Delete the `if backend_name == "flux1-kontext" and stream_kind == "single_stream"` skip branch. |
| `havedit/flux/warp.py` | 451 | Delete the same skip branch in `enable_havedit`. |
| `havedit/flux/warp.py` | 79-108 | Delete the `if backend_name == "flux1-kontext"` if-branch inside `_prepare_condition_images`; behavior moves into `Flux1KontextBackendAdapter.prepare_condition_images`. |

> All three deleted branches are already gated by `if backend_name == "flux1-kontext"`. For `flux2` runs the condition is always False and the body is dead code, so deletion is a strict no-op for the FLUX.2 path.

### Unchanged

- `havedit/backends/__init__.py` — keeps the existing `"flux1-kontext"` registration that already points to the new module path.
- `havedit/flux/head_scores.py`, `semantic_prior.py`, `boundary_consistency.py`, `subject_mask_headwise.py`, `runtime.py`, `config.py`, `trajectory_trust.py` — used by both backends; not modified.
- All FLUX.2 paths (`backends/flux2.py`, the existing attention processor, `warp.py` outside the listed deletions, `scripts/run_piebench_batch.py`, all other scripts/tests) — 0 functional change.

## 4. New Attention Processor

**Module:** `havedit/flux/attention_processor_flux1_kontext.py`

**Class:** `FluxKontextHAVAttnProcessor(state, block_name)`

**Architecture model:** Built on top of diffusers `FluxAttnProcessor`
(`transformer_flux.py:75`). Uses `to_q/to_k/to_v`, `add_q_proj/add_k_proj/add_v_proj`,
`norm_q/norm_k`, `norm_added_q/norm_added_k`, and `apply_rotary_emb` from
diffusers. This is intentionally a different API surface than FLUX.2's attention
processor — the two paths share zero attention-processor code.

**Token layout** (post text+image concatenation, same convention as KV-Edit / RegionE):

```
[ text tokens | pred-latent tokens | ref-image tokens ]
  0          text_n               text_n+latent_n     text_n+latent_n+ref_n
```

**`__call__` flow:**

1. Project Q/K/V from `hidden_states` (image-only, contains pred + ref) using `to_q/to_k/to_v`.
2. Project Q/K/V from `encoder_hidden_states` (text) using `add_q_proj/add_k_proj/add_v_proj` if `attn.added_kv_proj_dim is not None`.
3. Apply `norm_q/norm_k/norm_added_q/norm_added_k` per diffusers convention.
4. Concatenate to full sequence `[text | image]` along sequence dim.
5. Apply RoPE via `apply_rotary_emb(..., sequence_dim=1)`.
6. If `state.current_step < config.wsp.warmup_steps`: materialize text-query × all-keys attention (`O(text_n × seq_len)` not `O(seq_len²)`), softmax, extract text→image block, feed into `state.semantic_accumulator.update(...)`.
7. If past warmup:
   - Split `value` into `v_pred = value[:, :, text_n:text_n+latent_n, :]` and `v_ref = value[:, :, text_n+latent_n:text_n+latent_n+ref_n, :]`.
   - Compute `deviation = ||v_pred - v_ref||` per head per token.
   - Compute `z_dev = _local_zscore_map(deviation, height, width, kernel=havsr.local_kernel_size)`.
   - Compute `preserve_score = sigmoid(-α·z_dev + β·(1 - sem_prior))` using `config.havsr.alpha/beta` and the current semantic prior (running or final).
   - Soft preserve weight: `((score - threshold) / soft_band).clamp(0, 1)` with `soft_band = max((1 - threshold) * 0.5, eps)`.
   - At `current_step + 1 == background_discovery_step` and on the **last** double-stream block: freeze subject + background masks via `build_subject_candidate → refine_subject_candidate → select_subject_region → dilate_subject_region`.
   - At `current_step + 1 >= subject_release_step` and `release_scale < 1.0`: `preserve_weight[subject] *= release_scale`.
   - Combine: `v_pred_new = v_ref + (1 - preserve_weight).unsqueeze(-1) * (v_pred - v_ref)` (trust = 1 because trajectory_trust is excluded).
   - If `config.bhc.enabled`: apply `apply_boundary_head_consistency(v_pred_new, v_ref, preserve_score, ...)` using `config.bhc.tau_low/tau_high/lambda_max`.
   - Write the modified pred slice back into `value`.
8. Dispatch attention via `dispatch_attention_fn(query, key, value, attn_mask=attention_mask, ...)`.
9. Split output back into `encoder_hidden_states` (text) and `hidden_states` (image), apply `to_out` / `to_add_out`.
10. Return `(hidden_states, encoder_hidden_states)`.

**Reused utility imports** (pure functions, zero coupling):

```python
from havedit.flux.head_scores import _local_zscore_map, compute_head_deviation
from havedit.flux.semantic_prior import compute_step_semantic_prior, WarmupSemanticPrior
from havedit.flux.boundary_consistency import apply_boundary_head_consistency
from havedit.flux.subject_mask_headwise import (
    build_subject_candidate, refine_subject_candidate,
    select_subject_region, dilate_subject_region,
)
```

**Single-stream blocks are NOT hooked.** They retain the default `FluxAttnProcessor`
because (a) single-stream blocks don't have `add_q_proj` for ref-image projection
separation, and (b) the design hypothesis (validated in FLUX.2) is that double-stream
blocks carry the editing-relevant signal.

**Failure-mode handling:**

- `attention_mask is not None` → skip WSP accumulation (safety net; FLUX.1-Kontext's main path doesn't pass an attention mask).
- `v_pred.shape[2] == 0` or `v_pred.shape[2] != v_ref.shape[2]` → skip HAVSR, return original value.
- Semantic prior unavailable (both `latest_semantic_prior_current` and `semantic_prior` are None) → skip HAVSR, return original value. This happens during warmup before any accumulation has occurred.

## 5. Backend Adapter

**Module:** `havedit/backends/flux1_kontext.py`

**Class:** `Flux1KontextBackendAdapter`, implements the `BackendAdapter` protocol
defined in `havedit/backends/base.py`.

**Pipeline class:** `diffusers.pipelines.flux.pipeline_flux_kontext.FluxKontextPipeline`.

### Method differences from `Flux2BackendAdapter`

| Method | Behavior |
| --- | --- |
| `load_pipeline` | `FluxKontextPipeline.from_pretrained(runtime_cfg.model_path, torch_dtype=..., local_files_only=True, low_cpu_mem_usage=True)` then `configure_pipeline_runtime(pipe, runtime_cfg)`. |
| `iter_transformer_blocks` | Yields **only** `transformer_blocks` (double-stream) as `(block, f"ds_{i}", "double_stream")`. Single-stream blocks are intentionally not yielded so they keep their original processor across `enable_havedit` / `disable_havedit`. |
| `check_inputs` | Forwards to `pipeline.check_inputs(...)` with FLUX.1-Kontext's expected signature. |
| `build_run_context` | Calls `pipeline.encode_prompt(...)` which returns `(prompt_embeds, pooled_prompt_embeds, text_ids)`; calls `pipeline.prepare_latents(image=..., ...)` which returns `(latents, image_latents, latent_ids, image_ids)` — keeps `latent_ids` and `image_ids` as separate `BackendRunContext.latent_ids` / `reference_ids` fields (mirrors `Flux2BackendAdapter`). Note `image_ids[..., 0] = 1` is set by `prepare_latents` to distinguish reference-image tokens via RoPE. Populates `BackendRunContext.pooled_prompt_embeds`. |
| `build_step_context` | Cats `[latents, image_latents]` along sequence dim, cats `[latent_ids, reference_ids]` along sequence dim into `latent_image_ids`, builds `guidance = full(guidance_scale).expand(batch)`, includes `timestep`, `pooled_projections`. |
| `run_cond_step` | Calls `pipeline.transformer(hidden_states=..., timestep=t/1000, guidance=guidance, pooled_projections=pooled_prompt_embeds, encoder_hidden_states=prompt_embeds, txt_ids=text_ids, img_ids=latent_image_ids, joint_attention_kwargs=..., return_dict=False)[0]`. |
| `run_uncond_step` | Mirrors `run_cond_step` with negative embeddings. Only invoked if `should_run_uncond_step` returns True. |
| `should_run_uncond_step` | Returns `false` unconditionally (we don't use `true_cfg_scale > 1` in this port). |
| `scheduler_step` | `pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]` — identical to `Flux2BackendAdapter`. |
| `decode_latents` | `_unpack_latents(latents, height, width, vae_scale_factor)` → `(latents / vae.config.scaling_factor) + vae.config.shift_factor` → `vae.decode(...)` → `image_processor.postprocess(image, output_type=output_type)`. No BN renorm; no `_unpack_latents_with_ids` / `_unpatchify_latents`. |
| `prepare_condition_images` | Migrated from `warp.py:65-108`. Snaps input image to nearest `PREFERRED_KONTEXT_RESOLUTIONS` (imported from `diffusers.pipelines.flux.pipeline_flux_kontext`), aligns to `pipeline.vae_scale_factor * 2`, calls `image_processor.resize` + `image_processor.preprocess`. |
| `build_attention_processor` | Returns `FluxKontextHAVAttnProcessor(state=state, block_name=block_name)`; ignores `default_processor_cls`. This hook is already supported by `warp.py:583-597`. |

### Run context fields populated

`BackendRunContext` from `base.py`:

- `latents`, `reference_latents`, `latent_ids`, `reference_ids`
- `prompt_embeds`, `pooled_prompt_embeds`, `text_ids`
- `negative_prompt_embeds = None`, `negative_pooled_prompt_embeds = None`, `negative_text_ids = None` (CFG off)
- `height`, `width`, `image_height`, `image_width`
- `text_n = prompt_embeds.shape[1]`, `latent_n = latents.shape[1]`, `ref_n = image_latents.shape[1] if image_latents is not None else 0`
- `do_true_cfg = False`

## 6. Ablation Orchestration

**Script:** `scripts/run_dual_backbone_ablation.sh` (bash).

Runs `python scripts/run_piebench_batch.py` six times. Each invocation:

- Sets `--backend` to `flux2` or `flux1-kontext`.
- Sets `--model-path` to the corresponding weight directory.
- Sets `--output-dir` to a per-cell subdirectory under `${OUTPUT_ROOT}` (default: `${REPO_ROOT}/output/ablation_dual_backbone`).
- Always passes `--disable-trajectory-trust`.
- Passes `--disable-havedit` for baseline cells, `--disable-bhc` for w/o-BHC cells, nothing extra for full cells.
- Forwards any user-provided extra args (e.g. `--limit 3` for smoke runs, `--seed 0`, `--sample-ids ...`).

The six output subdirectories are:

- `flux2_baseline`
- `flux2_havedit_no_bhc`
- `flux2_havedit_full`
- `flux1k_baseline`
- `flux1k_havedit_no_bhc`
- `flux1k_havedit_full`

Each contains the standard `run_piebench_batch.py` output: per-sample subdirs
with `original.jpg`, `strength_1_00.jpg`, `metadata.json`.

**Smoke run:** `bash scripts/run_dual_backbone_ablation.sh --limit 3` — runs 3
samples per cell to validate all 6 pipelines without crashes; ~15 minutes total.

**Full run:** `bash scripts/run_dual_backbone_ablation.sh` — runs the full
PIE-Bench split, overnight.

## 7. Testing

**File:** `tests/test_flux1_kontext_adapter.py` (~150 lines, mock-based, no model weights).

| Test | Verifies |
| --- | --- |
| `test_get_backend_adapter_class_returns_flux1_kontext` | `get_backend_adapter_class("flux1-kontext")` returns a class with `backend_name == "flux1-kontext"`. |
| `test_iter_transformer_blocks_yields_only_double_stream` | Given a mock pipeline with 3 double + 2 single blocks, exactly 3 blocks are yielded, all tagged `"double_stream"`. |
| `test_build_run_context_captures_pooled_embeds` | With a mock `pipeline.encode_prompt` returning a 3-tuple, `ctx.pooled_prompt_embeds` is populated; `ctx.text_n`, `ctx.latent_n`, `ctx.ref_n` reflect the mock tensors' shapes. |
| `test_prepare_condition_images_snaps_to_kontext_resolution` | Given an 800×1200 PIL image, output is resized to the nearest `PREFERRED_KONTEXT_RESOLUTIONS` candidate. |
| `test_processor_accumulates_semantic_prior_in_warmup` | Six `__call__` invocations with `state.current_step` in 0–5 increment `state.semantic_accumulator.count` to 6. |
| `test_processor_havsr_replaces_only_pred_values` | Post-warmup, with valid `state.semantic_prior`, only the `[text_n:text_n+latent_n]` slice of `value` is modified; the `[text_n+latent_n:]` slice equals input. |
| `test_processor_bhc_disabled_path_skips_application` | With `config.bhc.enabled = False`, `apply_boundary_head_consistency` is never invoked (use `unittest.mock.patch`). |

E2E (weights-required) testing is deferred to the smoke run in §6.

## 8. Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| diffusers version skew breaks `FluxKontextPipeline` API. | Unit tests mock the pipeline; the smoke run surfaces real-version bugs immediately. Pin diffusers version in `requirements.txt` if needed. |
| User accidentally enables `trajectory_trust`. | Ablation shell script hardcodes `--disable-trajectory-trust` for every cell. |
| FLUX.1-Kontext CFG path (`true_cfg_scale > 1`) needed later. | `should_run_uncond_step` is an adapter method; can be extended to honor `true_cfg_scale` in a follow-up. |
| Semantic prior is None on first post-warmup steps. | Processor early-returns the original value (documented in §4 failure-mode handling). |
| `disable_havedit` is called but single-stream blocks weren't tracked. | Single-stream blocks are never modified, so there's nothing to restore. The `original_processors` list only contains double-stream entries. Verified by `test_iter_transformer_blocks_yields_only_double_stream`. |
| Output directories collide between cells. | Each cell writes to a distinct subdirectory under `${OUTPUT_ROOT}`. `--overwrite` is per-run, not cross-cell. |

## 9. Out of Scope

- Negative-prompt CFG support on FLUX.1-Kontext (`true_cfg_scale > 1`).
- Single-stream HAV processor on FLUX.1-Kontext.
- Metric aggregation across cells (PSNR/LPIPS/CLIP/VIEScore over the 6 directories). The user's existing `evaluation` tooling consumes the directory structure produced here.
- qwen-image-edit backend (still a stub in `_BACKENDS`).
- Re-validating `trajectory_trust` on FLUX.1-Kontext. It was excluded by author judgment.

## 10. Next Step

After this design is approved, invoke `superpowers:writing-plans` to produce the
task-by-task implementation plan.
