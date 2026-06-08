# HAV → SpotEdit-style K/V Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make HAVEdit's V/K-replacement on FLUX.1-Kontext-dev structurally identical to SpotEdit's K/V mixing, by replacing the current "clean-source v_ref" substitution target with a "cached K/V from a recent full forward" target — keeping HAV's per-head per-token gain (`1 - preserve_weight`) as the blending weight instead of SpotEdit's scalar `(1 - lmd)`.

**Architecture:** SpotEdit blends `(1-lmd) * current_K/V + lmd * cached_K/V` with scalar `lmd = cos²(πt/2)`. HAV blends `gain * v_pred + (1-gain) * v_ref` with per-head per-token `gain`. The two formulas are topologically identical; the only structural difference is the blending weight type. We've been using "v_ref from clean source latent at ref-slot" as the target, which is OOD on FLUX.1-Kontext. This plan switches the target to "cached K/V at pred-slot from a recent full forward" — the same source SpotEdit uses, which is noise-level-matched and model-friendly. HAV's per-head granularity, WSP, BHC, and Subject Release are all preserved.

**Tech Stack:** Python 3.10+, PyTorch 2.x, diffusers (FluxKontextPipeline, FluxAttention).

**Repo not under git:** `git add` / `git commit` steps omitted. WSL has no torch/pytest — tests run on GPU machine after handoff.

**Reference:** SpotEdit's blending formula (`fluxSpotAttn.py:96-97`):
```python
expanded_key   = (1 - lmd) * expanded_key   + lmd * self._cached_keys[:, text_n : text_n + latent_n2]
expanded_value = (1 - lmd) * expanded_value + lmd * self._cached_values[:, text_n : text_n + latent_n2]
```

HAV's blending formula (current `_maybe_replace_pred_kv`):
```python
v_pred_new = v_ref + gain * (v_pred - v_ref)  # ≡ gain * v_pred + (1 - gain) * v_ref
k_pred_new = k_ref + gain * (k_pred - k_ref)
```

The structural mapping:

| SpotEdit | HAV (new) |
|---|---|
| `expanded_key` (current K at this position) | `k_pred` (current K at pred position) |
| `self._cached_keys` (cached K from prior full forward) | `self._cached_k_pred` (cached K at pred position from warmup-end full forward) |
| scalar `(1 - lmd)` with `lmd = cos²(πt/2)` | per-head per-token `gain = 1 - preserve_weight` |

---

## File Structure

**Modified files:**

| Path | Change |
| --- | --- |
| `havedit/flux/attention_processor_flux1_kontext.py` | Add `_cached_k_pred` / `_cached_v_pred` state; capture at warmup end; use as substitution target in `_maybe_replace_pred_kv` |
| `havedit/backends/flux1_kontext.py` | Revert noise-the-reference logic (no longer needed; we're switching to SpotEdit-style cache instead) |
| `tests/test_flux1_kontext_adapter.py` | Update HAVSR / BHC tests to construct the cache state before calling `_maybe_replace_pred_kv` |

**Unmodified:**

- HAV math core (`head_scores.py`, `semantic_prior.py`, `boundary_consistency.py`, `subject_mask_headwise.py`)
- The blending formula in `_maybe_replace_pred_kv` (still `v_target + gain * (v_pred - v_target)`)
- WSP / Subject Release / BHC pipelines

---

## Task 1: Revert noise-the-reference logic from adapter

**Files:**
- Modify: `havedit/backends/flux1_kontext.py`

- [ ] **Step 1: Remove `ref_noise` sampling from `build_run_context`**

In `havedit/backends/flux1_kontext.py`, find the `build_run_context` method. Locate:

```python
        # Sample once-per-generation fixed noise used for post-warmup reference
        # latent noising. See `build_step_context` for the schedule. Using a fixed
        # noise tensor (same across all denoising steps) keeps the trajectory
        # deterministic and matches KV-Edit_inf's `z_t_src = (1-t)·src + t·ε`
        # construction where ε is fixed.
        ref_noise = None
        if image_latents is not None:
            ref_noise = torch.randn(
                image_latents.shape,
                device=image_latents.device,
                dtype=image_latents.dtype,
            )
```

Delete this entire block (the comment and the if/else assignment).

Also find and delete:

```python
        # Attach the fixed reference noise to run_ctx (dataclass allows extra attrs).
        # build_step_context will use it to construct noised_ref per step.
        run_ctx.ref_noise = ref_noise
```

Keep `return run_ctx` after `run_ctx = BackendRunContext(...)`.

- [ ] **Step 2: Revert `build_step_context` to use clean reference**

In the same file, find the `build_step_context` method. Replace this block:

```python
        if run_ctx.reference_latents is not None:
            # === Post-warmup reference noising ===
            # During warmup, feed clean reference latent so the WSP semantic prior
            # accumulates from the model's natural attention on the source image
            # (same as RegionE / SpotEdit do during their warmup).
            #
            # After warmup, HAVSR substitutes v_pred toward v_ref via
            # `v_pred_new = v_ref + gain * (v_pred - v_ref)`. This linear
            # interpolation is only geometrically meaningful when v_ref and v_pred
            # live in the same distribution. Since v_pred is at noise level σ_t,
            # we must put v_ref at the same σ_t. We achieve this by noising the
            # reference latent itself before it goes through the model:
            #     noised_ref_t = (1 - σ_t) · source + σ_t · ε_fixed
            # Then to_v(modulated(noised_ref_t)) lands in σ_t's distribution,
            # matching v_pred. This mirrors KV-Edit_inf's z_t_src construction.
            ref_for_model = run_ctx.reference_latents
            state = getattr(pipeline, "_havedit_state", None)
            ref_noise = getattr(run_ctx, "ref_noise", None)
            if state is not None and ref_noise is not None:
                warmup_steps = int(state.config.wsp.warmup_steps)
                current_step = int(state.current_step)
                if current_step >= warmup_steps:
                    sigmas = getattr(pipeline.scheduler, "sigmas", None)
                    if sigmas is not None and current_step < sigmas.numel():
                        sigma_t = float(sigmas[current_step].item())
                        sigma_t = max(0.0, min(1.0, sigma_t))
                        ref_for_model = (1.0 - sigma_t) * run_ctx.reference_latents + sigma_t * ref_noise

            latent_model_input = torch.cat(
                [latents, ref_for_model], dim=1
            ).to(pipeline.transformer.dtype)
            # FluxKontext ids are 2D (seq, 3); concat along seq dim = dim 0.
            latent_image_ids = torch.cat(
                [run_ctx.latent_ids, run_ctx.reference_ids], dim=0
            )
```

with the original simpler version:

```python
        if run_ctx.reference_latents is not None:
            latent_model_input = torch.cat(
                [latents, run_ctx.reference_latents], dim=1
            ).to(pipeline.transformer.dtype)
            # FluxKontext ids are 2D (seq, 3); concat along seq dim = dim 0.
            latent_image_ids = torch.cat(
                [run_ctx.latent_ids, run_ctx.reference_ids], dim=0
            )
```

- [ ] **Step 3: Remove unused `torch` import if it became unused**

Check if `import torch` at the top of `flux1_kontext.py` is still used elsewhere. (It probably still is — `build_step_context` uses `torch.cat` and `torch.full`.) Leave it if still used.

Run: `grep "torch\." /mnt/c/code/HAVEdit-main_changed/havedit/backends/flux1_kontext.py | head -5`
Expected: matches in `build_step_context` (`torch.cat`, `torch.full`). Keep the import.

---

## Task 2: Add K/V cache state to processor

**Files:**
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`

- [ ] **Step 1: Add cache attributes to `__init__`**

Find the `FluxKontextHAVAttnProcessor.__init__` method. Replace:

```python
    def __init__(self, state, block_name: str, is_single_stream: bool = False):
        self.state = state
        self.block_name = block_name
        self.is_single_stream = is_single_stream
```

with:

```python
    def __init__(self, state, block_name: str, is_single_stream: bool = False):
        self.state = state
        self.block_name = block_name
        self.is_single_stream = is_single_stream
        # SpotEdit-style K/V cache. Captured at the end of warmup (the last step
        # where HAV is OFF and the model runs naturally). Holds the pred-slot K and
        # V projections (post-norm, pre-RoPE) at that step. Subsequent steps use
        # these as the "pull toward" target in HAV's substitution:
        #     v_pred_new = (1 - gain) * v_target + gain * v_pred
        # where v_target = self._cached_v_pred and gain = 1 - preserve_weight.
        # This makes the target a real model-output K/V (noise-level matched, in
        # distribution), instead of clean-source projection (OOD on FLUX.1-Kontext).
        # Matches SpotEdit's `(1 - lmd) * current + lmd * cached_keys` structure
        # with gain replacing the scalar (1 - lmd).
        self._cached_k_pred = None
        self._cached_v_pred = None
```

- [ ] **Step 2: Verify by reading the file**

Run: `grep -nE "_cached_k_pred|_cached_v_pred" /mnt/c/code/HAVEdit-main_changed/havedit/flux/attention_processor_flux1_kontext.py | head -5`
Expected: matches at line ~80 (in __init__). No other matches yet (Task 3 adds them).

---

## Task 3: Capture cache at warmup end

**Files:**
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`

- [ ] **Step 1: Insert cache capture before the warmup gate**

In `_maybe_replace_pred_kv`, find:

```python
        if self.state.current_step < self.state.config.wsp.warmup_steps:
            return key, value
        if self.state.text_n <= 0 or self.state.latent_n <= 0 or self.state.ref_n <= 0:
            return key, value
        if value.shape[1] < self.state.text_n + self.state.latent_n + self.state.ref_n:
            return key, value

        key_bh = key.transpose(1, 2)      # [B, H, total_seq, D]
        value_bh = value.transpose(1, 2)
        v_pred, v_ref, pred_start, pred_end = self._split_pred_ref(value_bh)
        k_pred, k_ref, _, _ = self._split_pred_ref(key_bh)
        if v_pred.shape[2] == 0 or v_pred.shape[2] != v_ref.shape[2]:
            return key, value
```

Replace with:

```python
        if self.state.text_n <= 0 or self.state.latent_n <= 0 or self.state.ref_n <= 0:
            return key, value
        if value.shape[1] < self.state.text_n + self.state.latent_n + self.state.ref_n:
            return key, value

        key_bh = key.transpose(1, 2)      # [B, H, total_seq, D]
        value_bh = value.transpose(1, 2)
        v_pred, v_ref, pred_start, pred_end = self._split_pred_ref(value_bh)
        k_pred, k_ref, _, _ = self._split_pred_ref(key_bh)
        if v_pred.shape[2] == 0 or v_pred.shape[2] != v_ref.shape[2]:
            return key, value

        # Reset cache at the start of a new generation. HAVEditRuntimeState.begin_run
        # resets state but doesn't notify processor instances, so we detect via
        # current_step == 0.
        if self.state.current_step == 0:
            self._cached_k_pred = None
            self._cached_v_pred = None

        # Capture pred-slot K/V at the LAST warmup step (current_step == warmup_steps - 1).
        # At this point HAV is still OFF, so the K/V come from a natural full forward.
        # They live at noise level σ_{warmup_end} and are real model outputs — the
        # same property that lets SpotEdit's cache work on FLUX.1-Kontext-dev.
        warmup_end = self.state.config.wsp.warmup_steps - 1
        if self.state.current_step == warmup_end:
            self._cached_k_pred = k_pred.detach().clone()
            self._cached_v_pred = v_pred.detach().clone()

        # During warmup (and the capture step itself), HAV does NOT substitute.
        if self.state.current_step < self.state.config.wsp.warmup_steps:
            return key, value
```

Note: this restructures the gate so the cache capture runs BEFORE the warmup early-return.

- [ ] **Step 2: Verify the order of operations**

Run: `grep -nE "current_step == warmup_end|current_step < self.state.config.wsp|_cached_k_pred = " /mnt/c/code/HAVEdit-main_changed/havedit/flux/attention_processor_flux1_kontext.py`
Expected:
- Capture line `if self.state.current_step == warmup_end:` appears BEFORE the warmup gate `if self.state.current_step < ...`
- `_cached_k_pred = None` at one line (reset at step 0)
- `_cached_k_pred = k_pred.detach().clone()` at one line (capture at warmup end)

---

## Task 4: Use cached K/V as substitution target

**Files:**
- Modify: `havedit/flux/attention_processor_flux1_kontext.py`

- [ ] **Step 1: Inject the cached K/V into the substitution path**

Key design decision: **detection** uses `v_ref` (clean source V, drives content discrimination), **substitution** uses `v_target` (cached V, in-distribution mixing). These are different roles.

- `_compute_preserve_weight(v_pred, v_ref)` measures `||v_pred - v_ref||`. The deviation magnitude is what differentiates "edit region" (large deviation, low preserve_weight) from "background region" (small deviation, high preserve_weight). This signal still works even though v_ref is OOD for direct injection — we're just measuring distance.
- The actual substitution `v_target + gain * (v_pred - v_target)` uses `v_target` (cached) so that the resulting V is in-distribution.

In `_maybe_replace_pred_kv`, find:

```python
        self._maybe_freeze_subject_mask()
        weight_and_score = self._compute_preserve_weight(v_pred, v_ref)
        if weight_and_score is None:
            return key, value
        preserve_weight, _preserve_score = weight_and_score

        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)

        # Symmetric replacement: same gain on K and V keeps attention self-consistent.
        v_pred_new = v_ref + gain * (v_pred - v_ref)
        k_pred_new = k_ref + gain * (k_pred - k_ref)
```

Replace with:

```python
        # SpotEdit-style substitution target: cached pred-slot K/V from warmup_end.
        # These are real model-output projections at noise level σ_{warmup_end},
        # in-distribution and model-friendly. Falls back to the live ref-slot K/V
        # only if the cache is missing.
        if self._cached_k_pred is not None and self._cached_v_pred is not None:
            v_target = self._cached_v_pred.to(device=v_pred.device, dtype=v_pred.dtype)
            k_target = self._cached_k_pred.to(device=k_pred.device, dtype=k_pred.dtype)
        else:
            v_target = v_ref
            k_target = k_ref

        self._maybe_freeze_subject_mask()
        # IMPORTANT: preserve_weight detection still uses v_ref (clean source V
        # projection at this step). The deviation ||v_pred - v_ref|| is HAV's
        # signal for distinguishing "edit region" (large deviation, low
        # preserve_weight) from "background region" (small deviation, high
        # preserve_weight). We're only MEASURING here — the OOD-ness of v_ref
        # doesn't matter for measuring distance, only for direct injection.
        weight_and_score = self._compute_preserve_weight(v_pred, v_ref)
        if weight_and_score is None:
            return key, value
        preserve_weight, _preserve_score = weight_and_score

        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)

        # HAV's blending formula, structurally identical to SpotEdit:
        #   SpotEdit:  (1 - lmd) * expanded_key + lmd * cached_keys
        #   HAV (now): gain * v_pred + (1 - gain) * v_target
        # Per-head per-token gain replaces SpotEdit's scalar lmd. v_target (cached,
        # in-distribution) replaces v_ref (clean source, OOD) as the substitution
        # target — but preserve_weight (which determines gain) was computed
        # against v_ref, preserving HAV's content discrimination.
        v_pred_new = v_target + gain * (v_pred - v_target)
        k_pred_new = k_target + gain * (k_pred - k_target)
```

- [ ] **Step 2: Update the BHC `refs=` argument**

A few lines below, find the BHC block:

```python
        if self.state.config.bhc.enabled:
            v_pred_new = apply_boundary_head_consistency(
                values=v_pred_new,
                refs=v_ref,
                scores=_preserve_score,
                tau_low=self.state.config.bhc.tau_low,
                tau_high=self.state.config.bhc.tau_high,
                lambda_max=self.state.config.bhc.lambda_max,
            )
            k_pred_new = apply_boundary_head_consistency(
                values=k_pred_new,
                refs=k_ref,
                scores=_preserve_score,
                tau_low=self.state.config.bhc.tau_low,
                tau_high=self.state.config.bhc.tau_high,
                lambda_max=self.state.config.bhc.lambda_max,
            )
```

Replace `refs=v_ref` with `refs=v_target` and `refs=k_ref` with `refs=k_target` to keep BHC consistent with the new substitution target:

```python
        if self.state.config.bhc.enabled:
            v_pred_new = apply_boundary_head_consistency(
                values=v_pred_new,
                refs=v_target,
                scores=_preserve_score,
                tau_low=self.state.config.bhc.tau_low,
                tau_high=self.state.config.bhc.tau_high,
                lambda_max=self.state.config.bhc.lambda_max,
            )
            k_pred_new = apply_boundary_head_consistency(
                values=k_pred_new,
                refs=k_target,
                scores=_preserve_score,
                tau_low=self.state.config.bhc.tau_low,
                tau_high=self.state.config.bhc.tau_high,
                lambda_max=self.state.config.bhc.lambda_max,
            )
```

- [ ] **Step 3: Update the method docstring**

Find the docstring at the top of `_maybe_replace_pred_kv` and replace it with:

```python
        """Symmetric K + V substitution on the pred slice using SpotEdit-style cache.

        key / value shape: [B, total_seq, H, D]. Returns same shapes with the pred
        slice ([text_n : text_n + latent_n]) blended toward a cached K/V target.

        Structural equivalence to SpotEdit (`fluxSpotAttn.py:96-97`):
            SpotEdit:   expanded_key = (1 - lmd) * expanded_key + lmd * cached_keys
            HAV (this): k_pred_new   = (1 - gain) * k_target    + gain * k_pred
                      = k_target + gain * (k_pred - k_target)
            ↔ map: (1 - gain) ≡ lmd, k_pred ≡ expanded_key, k_target ≡ cached_keys

        HAV's only structural innovation over SpotEdit's mixing is the blending
        weight: scalar `lmd = cos²(πt/2)` is replaced by per-head per-token
        `gain = 1 - preserve_weight`, where preserve_weight comes from HAV's
        local z-score + WSP semantic prior pipeline.

        Cache source: captured at warmup_end (the last warmup step, where HAV is
        still OFF). At that point K/V are real model-output projections at
        noise level σ_{warmup_end} — in-distribution and model-friendly, which is
        why SpotEdit-style caching works on FLUX.1-Kontext-dev.
        """
```

- [ ] **Step 4: Verify the substitution uses cache**

Run: `grep -nE "v_target|k_target|_cached_k_pred|_cached_v_pred" /mnt/c/code/HAVEdit-main_changed/havedit/flux/attention_processor_flux1_kontext.py | head -15`

Expected: matches showing
- `_cached_k_pred = None` / `_cached_v_pred = None` (init + reset)
- `_cached_k_pred = k_pred.detach().clone()` (capture)
- `v_target = self._cached_v_pred.to(...)` (use)
- `k_target = self._cached_k_pred.to(...)` (use)
- `v_pred_new = v_target + gain * (v_pred - v_target)` (substitution)
- `k_pred_new = k_target + gain * (k_pred - k_target)` (substitution)
- `refs=v_target` and `refs=k_target` (BHC)

---

## Task 5: Update tests

**Files:**
- Modify: `tests/test_flux1_kontext_adapter.py`

The HAVSR and BHC tests currently call `_maybe_replace_pred_kv` directly with synthetic K/V tensors. They never run the warmup loop, so the cache (`_cached_k_pred` / `_cached_v_pred`) stays `None`, and the fallback `v_target = v_ref` path runs. This is fine for the existing behavioral assertions (pred slice modified, ref slice unchanged, BHC called once for K once for V) — but we should explicitly assert the cache fallback path works AND add a new test for the cached path.

- [ ] **Step 1: Add a test for the cached-target path**

Append to `tests/test_flux1_kontext_adapter.py` (at the end):

```python
def test_processor_uses_cached_kv_as_target_after_warmup():
    """When _cached_k_pred / _cached_v_pred are populated, HAV's substitution
    target is the cache, not the live ref slice. At gain=0 (forced via
    preserve_weight=1), the pred slice should equal the cached K/V exactly.
    """
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    torch.manual_seed(0)
    state = _make_state(text_n=4, latent_n=16, ref_n=16, height=4, width=4)
    # Set semantic_prior such that preserve_weight will be near 1 everywhere → gain near 0.
    # HAVSR formula: weight = clamp((sigmoid(-α·z + β·(1-sem)) - threshold)/soft_band, 0, 1).
    # With α=0, β=10 and sem=0, score = sigmoid(10) ≈ 1 > threshold = 0.9, so weight = 1.
    state.config.havsr.alpha = 0.0
    state.config.havsr.beta = 10.0
    state.config.bhc.enabled = False  # isolate the substitution path
    state.begin_step(state.config.wsp.warmup_steps)  # past warmup
    state.semantic_prior = torch.zeros(1, state.height, state.width)

    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")

    batch, heads, head_dim = 1, 2, 8
    total = state.text_n + state.latent_n + state.ref_n

    # Populate the cache with a known tensor (what would have been captured at warmup_end).
    cached_v_pred = torch.randn(batch, heads, state.latent_n, head_dim)
    cached_k_pred = torch.randn(batch, heads, state.latent_n, head_dim)
    processor._cached_v_pred = cached_v_pred.clone()
    processor._cached_k_pred = cached_k_pred.clone()

    key = torch.randn(batch, total, heads, head_dim)
    value = torch.randn(batch, total, heads, head_dim)

    modified_key, modified_value = processor._maybe_replace_pred_kv(key, value)

    # Pred slice should be pulled fully toward the cached K/V at gain≈0.
    pred_start = state.text_n
    pred_end = pred_start + state.latent_n
    # transpose convention inside _maybe_replace_pred_kv: value_bh = value.transpose(1, 2)
    # so pred slice in the [B, H, T, D] view is at [:, :, pred_start:pred_end, :].
    # The returned value/key have shape [B, T_total, H, D], i.e. original layout.
    # So the pred slice in the returned tensor is at [:, pred_start:pred_end, :, :].
    # That should equal cached_v_pred when permuted back to [B, T, H, D].
    cached_v_pred_in_layout = cached_v_pred.transpose(1, 2)  # [B, T, H, D]
    cached_k_pred_in_layout = cached_k_pred.transpose(1, 2)
    torch.testing.assert_close(
        modified_value[:, pred_start:pred_end, :, :],
        cached_v_pred_in_layout,
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        modified_key[:, pred_start:pred_end, :, :],
        cached_k_pred_in_layout,
        atol=1e-3,
        rtol=1e-3,
    )


def test_processor_captures_kv_at_warmup_end():
    """Walking the processor through warmup, the cache should be populated at
    current_step == warmup_steps - 1 and remain stable afterwards.
    """
    from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor

    torch.manual_seed(0)
    state = _make_state(text_n=4, latent_n=16, ref_n=16, height=4, width=4)
    processor = FluxKontextHAVAttnProcessor(state=state, block_name="ds_0")

    batch, heads, head_dim = 1, 2, 8
    total = state.text_n + state.latent_n + state.ref_n

    warmup_end = state.config.wsp.warmup_steps - 1
    key_at_warmup_end = None
    value_at_warmup_end = None

    for step in range(state.config.wsp.warmup_steps + 2):
        state.begin_step(step)
        key = torch.randn(batch, total, heads, head_dim)
        value = torch.randn(batch, total, heads, head_dim)
        if step == warmup_end:
            key_at_warmup_end = key.clone()
            value_at_warmup_end = value.clone()
        processor._maybe_replace_pred_kv(key, value)

    pred_start = state.text_n
    pred_end = pred_start + state.latent_n
    # Cache should hold the pred slice K/V from warmup_end in [B, H, T, D] layout.
    expected_v_pred = value_at_warmup_end.transpose(1, 2)[:, :, pred_start:pred_end, :]
    expected_k_pred = key_at_warmup_end.transpose(1, 2)[:, :, pred_start:pred_end, :]
    torch.testing.assert_close(processor._cached_v_pred, expected_v_pred)
    torch.testing.assert_close(processor._cached_k_pred, expected_k_pred)
```

- [ ] **Step 2: Verify**

Run: `grep -c "^def test_" /mnt/c/code/HAVEdit-main_changed/tests/test_flux1_kontext_adapter.py`
Expected: 11 (was 9, added 2 new tests).

---

## Task 6: Smoke test guidance (manual, GPU machine only)

**Files:**
- None modified.

These are run-by-hand commands the user executes on the GPU host. No file changes.

- [ ] **Step 1: Run unit tests on GPU host**

```
cd <havedit-repo-on-gpu-host>
python -m pytest tests/test_flux1_kontext_adapter.py -v
```

Expected: 11 PASS. If any FAIL, paste traceback for diagnosis.

- [ ] **Step 2: gain≈0 sanity test on FLUX.1-Kontext**

```
python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev \
    --pie-root /data/ll/data/PIEbench \
    --output-dir output/smoke_flux1k_spotedit_style \
    --disable-trajectory-trust \
    --alpha 0.0 --beta 10.0 \
    --limit 1
```

Expected: output should be a clean image (no noise spots, no global color tint, no washed-out look). The semantics: at gain≈0 with SpotEdit-style cache, the pred slice's K/V are pulled fully toward the cache captured at warmup_end. This produces a "trajectory anchored at warmup-end" output — visually similar to the model's natural prediction at warmup_end, NOT the source image. That's expected: this is the same kind of "anchor to recent forward" semantics SpotEdit uses; it's NOT source-overwriting.

If the output is noisy: cache wasn't captured correctly. Check `processor._cached_v_pred is None` at any point post-warmup.
If the output is clean but doesn't match warmup-end prediction: substitution math has a bug.

- [ ] **Step 3: Default-config edit test on FLUX.1-Kontext**

```
python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev \
    --pie-root /data/ll/data/PIEbench \
    --output-dir output/smoke_flux1k_default \
    --disable-trajectory-trust \
    --limit 1
```

Expected: editing should happen as instructed AND background should be more preserved than the `--disable-havedit` baseline. This is what the paper ablation cares about.

- [ ] **Step 4: 6-cell ablation matrix**

```
bash scripts/run_dual_backbone_ablation.sh --limit 3
```

Expected: all 6 cells (2 backbones × 3 ablation configs) finish without crashing. Each output cell has 3 sample subdirs with `strength_1_00.jpg` + `metadata.json`.

---

## Spec Coverage Check

| Spec section | Implementing task |
| --- | --- |
| Revert noise-the-reference (adapter) | Task 1 |
| Add `_cached_k_pred` / `_cached_v_pred` state | Task 2 |
| Capture cache at warmup_end | Task 3 |
| Use cache as substitution target (HAV gain stays) | Task 4 |
| Update BHC to use cached target | Task 4 Step 2 |
| Update docstring | Task 4 Step 3 |
| Unit tests for cached-target path | Task 5 |
| Unit test for cache-capture timing | Task 5 |
| Smoke-test guidance | Task 6 |

---

## Notes for the Implementer

1. **HAV math formula is preserved.** The blend `v_pred_new = v_target + gain * (v_pred - v_target)` is the exact same formula as before — only `v_target` changed from `v_ref` (clean source) to `self._cached_v_pred` (cached at warmup_end).

2. **At gain=0, the output won't be the source image.** It will be visually similar to whatever the model would have produced if denoising stopped at warmup_end. This is the SAME behavior as SpotEdit's K/V mixing — SpotEdit gets "source image" output by adding a separate latent-overwrite step at the end of the pipeline (`latents[:, cache_final] = image_latents[:, cache_final]`), which the user has explicitly excluded from this plan. The K/V mixing alone is NOT a source-overwrite mechanism.

3. **What this fixes:** the K/V being injected are now real model outputs (in-distribution), so HAV's substitution stops producing noise / color tints / washed-out artifacts. The per-head per-token gain still drives where the substitution acts strongly vs weakly.

4. **What this does NOT fix:** "gain=0 → source image" is a property HAV does not have at all. The user has confirmed they understand this is not the goal — the goal is correct K/V injection so HAV can operate as designed in its soft-weight regime.
