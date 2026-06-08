from __future__ import annotations

import torch
from diffusers import DiffusionPipeline

from lsedit.backends.base import BackendRunContext
from lsedit.flux.model_loader import configure_pipeline_runtime, resolve_dtype


def _iter_transformer_blocks(transformer):
    for index, block in enumerate(getattr(transformer, "transformer_blocks", [])):
        yield block, f"ds_{index}", "double_stream"
    for index, block in enumerate(getattr(transformer, "single_transformer_blocks", [])):
        yield block, f"ss_{index}", "single_stream"


class Flux2BackendAdapter:
    backend_name = "flux2"

    @staticmethod
    def _spatial_shape_from_ids(token_ids, token_count: int) -> tuple[int, int]:
        if token_ids is None or getattr(token_ids, "numel", lambda: 0)() == 0 or token_ids.shape[-1] < 3:
            return 1, token_count

        height = int(token_ids[..., 1].max().item()) + 1
        width = int(token_ids[..., 2].max().item()) + 1
        return height, width

    def load_pipeline(self, runtime_cfg):
        pipe = DiffusionPipeline.from_pretrained(
            runtime_cfg.model_path,
            torch_dtype=resolve_dtype(runtime_cfg.torch_dtype),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        return configure_pipeline_runtime(pipe, runtime_cfg)

    def iter_transformer_blocks(self, pipeline):
        yield from _iter_transformer_blocks(pipeline.transformer)

    def check_inputs(self, pipeline, params, callback_on_step_end_tensor_inputs):
        pipeline.check_inputs(
            prompt=params.prompt,
            height=params.height,
            width=params.width,
            prompt_embeds=params.prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=params.guidance_scale,
        )

    def build_run_context(self, pipeline, params):
        device = getattr(pipeline, "_execution_device", None)
        prompt = params.prompt
        prompt_embeds = params.prompt_embeds
        negative_prompt_embeds = params.negative_prompt_embeds
        num_images_per_prompt = params.num_images_per_prompt
        max_sequence_length = params.max_sequence_length
        text_encoder_out_layers = getattr(params, "text_encoder_out_layers", None)

        prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            text_encoder_out_layers=text_encoder_out_layers,
        )

        negative_text_ids = None
        if pipeline.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = pipeline.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        num_channels_latents = pipeline.transformer.config.in_channels // 4
        latents, latent_ids = pipeline.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=params.height,
            width=params.width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=params.generator,
            latents=params.latents,
        )

        image_latents = None
        image_latent_ids = None
        if params.condition_images is not None:
            image_dtype = getattr(getattr(pipeline, "vae", None), "dtype", prompt_embeds.dtype)
            image_latents, image_latent_ids = pipeline.prepare_image_latents(
                images=params.condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=params.generator,
                device=device,
                dtype=image_dtype,
            )

        latent_height, latent_width = self._spatial_shape_from_ids(latent_ids, latents.shape[1])
        return BackendRunContext(
            latents=latents,
            reference_latents=image_latents,
            latent_ids=latent_ids,
            reference_ids=image_latent_ids,
            prompt_embeds=prompt_embeds,
            text_ids=text_ids,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_text_ids=negative_text_ids,
            height=latent_height,
            width=latent_width,
            image_height=params.height,
            image_width=params.width,
            text_n=prompt_embeds.shape[1],
            latent_n=latents.shape[1],
            ref_n=0 if image_latents is None else image_latents.shape[1],
        )

    def build_step_context(self, pipeline, run_ctx, latents, timestep, guidance_scale):
        latent_model_input = latents.to(pipeline.transformer.dtype)
        latent_image_ids = run_ctx.latent_ids
        if run_ctx.reference_latents is not None:
            latent_model_input = torch.cat([latents, run_ctx.reference_latents], dim=1).to(pipeline.transformer.dtype)
            latent_image_ids = torch.cat([run_ctx.latent_ids, run_ctx.reference_ids], dim=1)
        return {
            "latent_model_input": latent_model_input,
            "latent_image_ids": latent_image_ids,
            "timestep": timestep.expand(latents.shape[0]).to(latents.dtype),
            "guidance_scale": guidance_scale,
        }

    def run_cond_step(self, pipeline, state, step_ctx):
        run_ctx = state
        with pipeline.transformer.cache_context("cond"):
            return pipeline.transformer(
                hidden_states=step_ctx["latent_model_input"],
                timestep=step_ctx["timestep"] / 1000,
                guidance=None,
                encoder_hidden_states=run_ctx.prompt_embeds,
                txt_ids=run_ctx.text_ids,
                img_ids=step_ctx["latent_image_ids"],
                joint_attention_kwargs=pipeline.attention_kwargs,
                return_dict=False,
            )[0]

    def run_uncond_step(self, pipeline, state, step_ctx):
        run_ctx = state
        with pipeline.transformer.cache_context("uncond"):
            return pipeline.transformer(
                hidden_states=step_ctx["latent_model_input"],
                timestep=step_ctx["timestep"] / 1000,
                guidance=None,
                encoder_hidden_states=run_ctx.negative_prompt_embeds,
                txt_ids=run_ctx.negative_text_ids,
                img_ids=step_ctx["latent_image_ids"],
                joint_attention_kwargs=pipeline._attention_kwargs,
                return_dict=False,
            )[0]

    def scheduler_step(self, pipeline, noise_pred, timestep, latents):
        return pipeline.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]

    def decode_latents(self, pipeline, latents, run_ctx, output_type, return_dict):
        if output_type == "latent":
            return latents

        vae = getattr(pipeline, "vae", None)
        image_processor = getattr(pipeline, "image_processor", None)
        if vae is None or image_processor is None:
            raise RuntimeError("FLUX.2 decode requires both vae and image_processor")
        if not callable(getattr(vae, "decode", None)) or not callable(getattr(image_processor, "postprocess", None)):
            raise RuntimeError("FLUX.2 decode requires callable vae.decode and image_processor.postprocess")

        unpack_with_ids = getattr(pipeline, "_unpack_latents_with_ids", None)
        unpack = getattr(pipeline, "_unpack_latents", None)
        unpatchify = getattr(pipeline, "_unpatchify_latents", None)

        latents_to_decode = latents
        if callable(unpack_with_ids):
            latents_to_decode = unpack_with_ids(latents_to_decode, run_ctx.latent_ids)

            bn = getattr(vae, "bn", None)
            bn_eps = getattr(getattr(vae, "config", None), "batch_norm_eps", None)
            if (
                bn is not None
                and isinstance(getattr(bn, "running_mean", None), torch.Tensor)
                and isinstance(getattr(bn, "running_var", None), torch.Tensor)
                and bn_eps is not None
            ):
                latents_bn_mean = bn.running_mean.view(1, -1, 1, 1).to(latents_to_decode.device, latents_to_decode.dtype)
                latents_bn_std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + bn_eps).to(
                    latents_to_decode.device, latents_to_decode.dtype
                )
                latents_to_decode = latents_to_decode * latents_bn_std + latents_bn_mean

            if not callable(unpatchify):
                raise RuntimeError("FLUX.2 decode with latent ids requires _unpatchify_latents")
            latents_to_decode = unpatchify(latents_to_decode)
        elif callable(unpack):
            latents_to_decode = unpack(latents_to_decode, run_ctx.image_height, run_ctx.image_width, pipeline.vae_scale_factor)
            latents_to_decode = (latents_to_decode / vae.config.scaling_factor) + vae.config.shift_factor
        else:
            raise RuntimeError("FLUX.2 decode requires either _unpack_latents_with_ids or _unpack_latents")

        image = vae.decode(latents_to_decode, return_dict=False)[0]
        return image_processor.postprocess(image, output_type=output_type)
