from __future__ import annotations

from typing import Any

import torch

from havedit.backends.base import BackendRunContext


class Flux1KontextBackendAdapter:
    backend_name = "flux1-kontext"

    def load_pipeline(self, runtime_cfg: Any) -> Any:
        from diffusers.pipelines.flux.pipeline_flux_kontext import FluxKontextPipeline
        from havedit.flux.model_loader import configure_pipeline_runtime, resolve_dtype

        pipe = FluxKontextPipeline.from_pretrained(
            runtime_cfg.model_path,
            torch_dtype=resolve_dtype(runtime_cfg.torch_dtype),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        return configure_pipeline_runtime(pipe, runtime_cfg)

    def iter_transformer_blocks(self, pipeline: Any):
        transformer = getattr(pipeline, "transformer", pipeline)
        for index, block in enumerate(getattr(transformer, "transformer_blocks", [])):
            yield block, f"ds_{index}", "double_stream"
        # NOTE: single_transformer_blocks intentionally NOT yielded. FLUX.1-Kontext
        # single-stream blocks retain their default FluxAttnProcessor.

    def check_inputs(self, pipeline: Any, params: Any, callback_on_step_end_tensor_inputs: Any) -> None:
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

    def build_run_context(self, pipeline, params):
        device = getattr(pipeline, "_execution_device", None)

        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=params.prompt,
            prompt_2=None,
            prompt_embeds=params.prompt_embeds,
            # warp.py's _wrapped_pipeline_class doesn't pack pooled_prompt_embeds
            # into the SimpleNamespace (it serves FLUX.2 which doesn't need pooled).
            # Use getattr fallback so this adapter works without warp.py changes.
            pooled_prompt_embeds=getattr(params, "pooled_prompt_embeds", None),
            device=device,
            num_images_per_prompt=params.num_images_per_prompt,
            max_sequence_length=params.max_sequence_length,
            lora_scale=None,
        )

        negative_prompt_embeds = None
        negative_pooled = None
        negative_text_ids = None
        # CFG (true_cfg_scale > 1) is intentionally not supported in this port.

        # condition_images may be:
        #   - None
        #   - A list of PIL/np images (when prepare_condition_images was bypassed)
        #   - A preprocessed Tensor of shape (B, 3, H, W) from our prepare_condition_images
        # FluxKontextPipeline.prepare_latents wants a single image-or-tensor; never index
        # a Tensor with [0] (would drop batch dim and break VAE encode).
        condition = params.condition_images
        if condition is None:
            condition_image = None
        elif isinstance(condition, list):
            condition_image = condition[0] if len(condition) > 0 else None
        else:
            condition_image = condition  # already a Tensor

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = pipeline.prepare_latents(
            image=condition_image,
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

        run_ctx = BackendRunContext(
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
        return run_ctx

    @staticmethod
    def _spatial_shape_from_ids(token_ids, token_count):
        if token_ids is None or getattr(token_ids, "numel", lambda: 0)() == 0 or token_ids.shape[-1] < 3:
            return 1, token_count
        height = int(token_ids[..., 1].max().item()) + 1
        width = int(token_ids[..., 2].max().item()) + 1
        return height, width

    def build_step_context(self, pipeline: Any, run_ctx: BackendRunContext, latents: Any, timestep: Any, guidance_scale: Any) -> Any:
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

    def run_cond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
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

    def run_uncond_step(self, pipeline: Any, state: Any, step_ctx: Any) -> Any:
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

    def should_run_uncond_step(self, pipeline: Any, run_ctx: BackendRunContext, guidance_scale: Any) -> bool:
        return False

    def scheduler_step(self, pipeline: Any, noise_pred: Any, timestep: Any, latents: Any) -> Any:
        return pipeline.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def decode_latents(self, pipeline: Any, latents: Any, run_ctx: BackendRunContext, output_type: Any, return_dict: Any) -> Any:
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

    def build_attention_processor(self, *, state, block_name, stream_kind, default_processor_cls):
        from havedit.flux.attention_processor_flux1_kontext import FluxKontextHAVAttnProcessor
        # By default `iter_transformer_blocks` only yields double-stream blocks, so
        # stream_kind is "double_stream" in production. The is_single_stream flag is
        # plumbed for parity with the FLUX.2 processor's dispatch pattern and so the
        # processor can correctly handle the single-stream attention layout if the
        # adapter is ever extended to attach there too.
        return FluxKontextHAVAttnProcessor(
            state=state,
            block_name=block_name,
            is_single_stream=(stream_kind == "single_stream"),
        )
