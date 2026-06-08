from __future__ import annotations

import torch
import torch.nn.functional as F

from lsedit.backends import get_backend_adapter_class

from .tbss import apply_tbss
from .lscp import _local_zscore_map, compute_head_deviation
from .runtime import LSEditRuntimeState
from .warp import _set_processor, _wrapped_pipeline_class

dispatch_attention_fn = None
apply_rotary_emb = None
_DIFFUSERS_HELPER_UNAVAILABLE = object()


def _load_dispatch_attention_fn():
    global dispatch_attention_fn
    if callable(dispatch_attention_fn):
        return dispatch_attention_fn
    if dispatch_attention_fn is _DIFFUSERS_HELPER_UNAVAILABLE:
        return None
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn as diffusers_dispatch_attention_fn
    except ImportError:
        dispatch_attention_fn = _DIFFUSERS_HELPER_UNAVAILABLE
        return None
    dispatch_attention_fn = diffusers_dispatch_attention_fn
    return dispatch_attention_fn


def _load_apply_rotary_emb():
    global apply_rotary_emb
    if callable(apply_rotary_emb):
        return apply_rotary_emb
    if apply_rotary_emb is _DIFFUSERS_HELPER_UNAVAILABLE:
        return None
    try:
        from diffusers.models.embeddings import apply_rotary_emb as diffusers_apply_rotary_emb
    except ImportError:
        apply_rotary_emb = _DIFFUSERS_HELPER_UNAVAILABLE
        return None
    apply_rotary_emb = diffusers_apply_rotary_emb
    return apply_rotary_emb


class FluxKontextHAVAttnProcessor:
    def __init__(self, state, block_name: str, is_single_stream: bool):
        self.state = state
        self.block_name = block_name
        self.is_single_stream = is_single_stream
        self._attention_backend = None
        self._parallel_config = None

    def __call__(self, attn, hidden_states: torch.Tensor, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None):
        if self.is_single_stream:
            attention_mask, image_rotary_emb = self._normalize_single_stream_inputs(
                encoder_hidden_states, attention_mask, image_rotary_emb
            )
            return self._call_single_stream(attn, hidden_states, attention_mask, image_rotary_emb)
        return self._call_double_stream(attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb)

    def _normalize_single_stream_inputs(self, encoder_hidden_states, attention_mask, image_rotary_emb):
        if isinstance(attention_mask, tuple):
            if image_rotary_emb is None:
                image_rotary_emb = attention_mask
            attention_mask = encoder_hidden_states if isinstance(encoder_hidden_states, torch.Tensor) else None
        return attention_mask, image_rotary_emb

    def _call_double_stream(self, attn, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
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
        query = self._maybe_apply_rotary_emb(query, image_rotary_emb)
        key = self._maybe_apply_rotary_emb(key, image_rotary_emb)
        self._maybe_accumulate_from_attention(query, key, attention_mask)
        value = self._maybe_replace_double_stream_values(value)
        hidden_states = self._dispatch_attention(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        return hidden_states

    def _call_single_stream(self, attn, hidden_states, attention_mask, image_rotary_emb):
        hidden_states = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        query = self._maybe_apply_rotary_emb(query, image_rotary_emb)
        key = self._maybe_apply_rotary_emb(key, image_rotary_emb)
        hidden_states = self._dispatch_attention(query, key, value, attention_mask)
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
        hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
        return attn.to_out(hidden_states)

    def _maybe_apply_rotary_emb(self, tensor, image_rotary_emb):
        if image_rotary_emb is None:
            return tensor
        rotary_emb = _load_apply_rotary_emb()
        if rotary_emb is None:
            raise ImportError("diffusers rotary embedding helpers are required when image_rotary_emb is provided")
        return rotary_emb(tensor, image_rotary_emb, sequence_dim=1)
    
    def _maybe_accumulate_from_attention(self, query, key, attention_mask):
        collect_semantic_prior = self.state.current_step < self.state.config.wsp.warmup_steps
        collect_attention_map = self.state.visualize_attention
        if attention_mask is not None or self.state.text_n <= 0 or self.state.latent_n <= 0:
            return
        if query.shape[1] < self.state.text_n + self.state.latent_n:
            return
        # The downstream diagnostics only consume text-query rows. Materializing
        # the full q x k matrix becomes prohibitively expensive for Kontext's
        # preferred resolutions, even though the core attention path remains fine.
        query_heads = query.transpose(1, 2)[:, :, : self.state.text_n, :].float()
        key_heads = key.transpose(1, 2).float()
        scale = query_heads.shape[-1] ** -0.5
        attn_scores = torch.matmul(query_heads, key_heads.transpose(-1, -2)) * scale
        attn_probs = attn_scores.softmax(dim=-1).to(query.dtype)
        self._maybe_update_semantic_prior_diagnostics(attn_probs)
        if collect_semantic_prior:
            self._maybe_accumulate_semantic_prior(attn_probs)
        if collect_attention_map:
            self._maybe_record_attention_map(attn_probs)

    def _dispatch_attention(self, query, key, value, attention_mask):
        dispatch_fn = _load_dispatch_attention_fn()
        if dispatch_fn is not None:
            return dispatch_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend, parallel_config=self._parallel_config)
        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        return hidden_states.transpose(1, 2)

    def _block_enabled(self) -> bool:
        scope = self.state.config.havsr.block_scope
        if scope == "double_stream":
            return not self.is_single_stream
        if scope == "single_stream":
            return self.is_single_stream
        return True

    def _lsedit_active(self) -> bool:
        end_step = int(getattr(self.state.config.runtime, "lsedit_end", 0) or 0)
        return end_step <= 0 or (self.state.current_step + 1) <= end_step

    def _split_pred_and_ref_values(self, values: torch.Tensor):
        pred_start = self.state.text_n
        pred_end = pred_start + self.state.latent_n
        ref_start = pred_end
        ref_end = ref_start + self.state.ref_n
        return values[:, :, pred_start:pred_end, :], values[:, :, ref_start:ref_end, :]


    def _maybe_update_semantic_prior_diagnostics(self, attn_probs: torch.Tensor) -> None:
        if hasattr(self.state, "update_semantic_prior_current"):
            self.state.update_semantic_prior_current(
                attn_probs=attn_probs,
                text_n=self.state.text_n,
                latent_n=self.state.latent_n,
            )

    def _maybe_accumulate_semantic_prior(self, attn_probs: torch.Tensor) -> None:
        if self.state.current_step < self.state.config.wsp.warmup_steps:
            self.state.semantic_accumulator.update(attn_probs, text_n=self.state.text_n, latent_n=self.state.latent_n)

    def _maybe_record_attention_map(self, attn_probs: torch.Tensor) -> None:
        if not self.state.visualize_attention or self.state.text_n <= 0 or self.state.latent_n <= 0:
            return
        block = attn_probs[:, :, : self.state.text_n, self.state.text_n : self.state.text_n + self.state.latent_n]
        spatial = block.mean(dim=(1, 2))
        if spatial.shape[-1] != self.state.height * self.state.width:
            return
        self.state.accumulate_attention_map(spatial.view(-1, self.state.height, self.state.width))


    def _maybe_replace_double_stream_values(self, value):
        if getattr(self.state, "capture_reference_values", False):
            text_n = self.state.text_n
            ref_n = self.state.ref_n
            if text_n > 0 and ref_n > 0 and value.shape[1] >= text_n + ref_n:
                value_by_head = value.transpose(1, 2)
                v_ref_capture = value_by_head[:, :, text_n:text_n + ref_n, :].detach()
                self.state.cache_reference_values(self.block_name, v_ref_capture)
            return value
        if self.state.current_step < self.state.config.wsp.warmup_steps:
            return value
        if self.state.current_step > self.state.config.hav_steps:
            return value
        if self.state.text_n <= 0 or self.state.latent_n <= 0 or self.state.ref_n <= 0:
            return value
        if value.shape[1] < self.state.text_n + self.state.latent_n + self.state.ref_n:
            return value
        value_by_head = value.transpose(1, 2)
        v_pred, v_ref = self._split_pred_and_ref_values(value_by_head)
        cached_ref = self.state.get_cached_reference_values(self.block_name)
        if cached_ref is not None:
            cached_ref = cached_ref.to(device=v_pred.device, dtype=v_pred.dtype)
        if v_pred.shape[2] == 0 or v_ref.shape[2] == 0 or v_pred.shape[2] != v_ref.shape[2]:
            return value
        replaced, scores = self._replace_latent_values_with_scores(v_pred, cached_ref)
        refined = self._apply_bhc(replaced, cached_ref, scores) if scores is not None else replaced
        pred_start = self.state.text_n
        pred_end = pred_start + v_pred.shape[2]
        value_by_head = value_by_head.clone()
        value_by_head[:, :, pred_start:pred_end, :] = refined
        return value_by_head.transpose(1, 2)

    def _apply_bhc(self, values: torch.Tensor, refs: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if not self._lsedit_active() or not self._block_enabled() or not self.state.config.bhc.enabled:
            return values
        return apply_tbss(
            values=values,
            refs=refs,
            scores=scores,
            tau_low=self.state.config.bhc.tau_low,
            tau_high=self.state.config.bhc.tau_high,
            lambda_max=self.state.config.bhc.lambda_max,
        )

    def _replace_latent_values_with_scores(
        self,
        v_pred: torch.Tensor,
        v_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self._block_enabled():
            return v_pred, None
        if v_pred.shape != v_ref.shape:
            raise ValueError("headwise routing requires v_pred and v_ref to have the same shape")

        semantic_prior = getattr(self.state, "running_semantic_prior", None)
        if semantic_prior is None:
            semantic_prior = getattr(self.state, "semantic_prior", None)
        if semantic_prior is None:
            return v_pred, None

        deviation = compute_head_deviation(v_pred, v_ref)
        z_dev = _local_zscore_map(
            deviation,
            height=self.state.height,
            width=self.state.width,
            kernel_size=self.state.config.havsr.local_kernel_size,
            eps=self.state.config.havsr.eps,
        )
        # if self.state.config.havsr.decision_granularity == "token":
        z_dev = z_dev.mean(dim=1, keepdim=True).expand_as(z_dev)
        sem = semantic_prior.reshape(semantic_prior.shape[0], 1, -1).to(dtype=z_dev.dtype, device=z_dev.device)
        preserve_score = torch.sigmoid(
            -float(self.state.config.havsr.alpha) * z_dev
            + float(self.state.config.havsr.beta) * (1.0 - sem)
        )
        # if self.state.config.havsr.decision_granularity == "token":
        preserve_score = preserve_score.mean(dim=1, keepdim=True).expand_as(preserve_score)
        threshold = float(self.state.config.havsr.threshold)
        soft_band = max((1.0 - threshold) * 0.5, float(self.state.config.havsr.eps))
        full_preserve_at = threshold + soft_band
        preserve_weight = ((preserve_score - threshold) / soft_band).clamp(0.0, 1.0)
        preserve_weight = torch.where(
            preserve_score >= full_preserve_at,
            torch.ones_like(preserve_weight),
            preserve_weight,
        )

        self.state.latest_headwise_zdev = z_dev.detach().to(dtype=torch.float32).cpu()
        self.state.latest_headwise_preserve_score = preserve_score.detach().to(dtype=torch.float32).cpu()
        self.state.latest_headwise_preserve_weight = preserve_weight.detach().to(dtype=torch.float32).cpu()
        token_preserve_weight = preserve_weight.detach().to(dtype=torch.float32).mean(dim=1)
        if token_preserve_weight.shape[-1] == self.state.height * self.state.width:
            preserve_weight_map = token_preserve_weight.view(
                token_preserve_weight.shape[0],
                self.state.height,
                self.state.width,
            )
            self.state.accumulate_preserve_weight_map(preserve_weight_map)
            self.state.accumulate_edit_mask(1.0 - preserve_weight_map)

        gain = (1.0 - preserve_weight).unsqueeze(-1).to(dtype=v_pred.dtype)
        return v_ref + gain * (v_pred - v_ref), preserve_score
