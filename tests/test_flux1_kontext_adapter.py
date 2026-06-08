from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from types import SimpleNamespace
from unittest.mock import MagicMock
from PIL import Image

from havedit.backends import get_backend_adapter_class
from havedit.backends.flux1_kontext import Flux1KontextBackendAdapter


def test_get_backend_adapter_class_returns_flux1_kontext():
    cls = get_backend_adapter_class("flux1-kontext")
    assert cls.backend_name == "flux1-kontext"


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
