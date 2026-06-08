from __future__ import annotations

import importlib
import inspect
from contextlib import contextmanager
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import torch

from havedit.backends import get_backend_adapter_class

from .pipeline import initialize_source_latents
from .runtime import HAVEditRuntimeState


def _set_processor(attention, processor) -> None:
    setter = getattr(attention, "set_processor", None)
    if callable(setter):
        setter(processor)
        return
    attention.processor = processor


def _iter_transformer_blocks(transformer):
    backend_name = getattr(transformer, "_havedit_backend", None)
    if backend_name is None:
        raise ValueError("Transformer is missing _havedit_backend")

    adapter = get_backend_adapter_class(backend_name)()
    yield from adapter.iter_transformer_blocks(SimpleNamespace(transformer=transformer))

def _spatial_shape_from_ids(token_ids, token_count: int) -> tuple[int, int]:
    if token_ids is None or getattr(token_ids, "numel", lambda: 0)() == 0 or token_ids.shape[-1] < 3:
        return 1, token_count

    height = int(token_ids[..., 1].max().item()) + 1
    width = int(token_ids[..., 2].max().item()) + 1
    return height, width


def _infer_image_size(image) -> tuple[int | None, int | None]:
    size = getattr(image, "size", None)
    if isinstance(size, tuple) and len(size) == 2:
        return int(size[0]), int(size[1])

    shape = getattr(image, "shape", None)
    if shape is None:
        return None, None

    shape = tuple(shape)
    if len(shape) == 2:
        return int(shape[1]), int(shape[0])
    if len(shape) == 3:
        if shape[0] in (1, 3, 4):
            return int(shape[2]), int(shape[1])
        return int(shape[1]), int(shape[0])
    if len(shape) == 4:
        if shape[1] in (1, 3, 4):
            return int(shape[3]), int(shape[2])
        return int(shape[2]), int(shape[1])

    return None, None


def _prepare_condition_images(pipeline, image, height, width):
    if image is None:
        return None, height, width

    backend_name = getattr(getattr(pipeline, "transformer", None), "_havedit_backend", None)
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

        image_width, image_height = _infer_image_size(img)
        if image_width is not None and image_height is not None:
            if image_width * image_height > 1024 * 1024:
                resize_to_target_area = getattr(image_processor, "_resize_to_target_area", None)
                if callable(resize_to_target_area):
                    img = resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = _infer_image_size(img)

            multiple_of = pipeline.vae_scale_factor * 2
            image_width = (image_width // multiple_of) * multiple_of
            image_height = (image_height // multiple_of) * multiple_of
            img = image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
            height = height or image_height
            width = width or image_width

        condition_images.append(img)

    return condition_images, height, width


def _prepare_timesteps(pipeline, base_module, latents: torch.Tensor, num_inference_steps: int, sigmas):
    retrieve_timesteps = getattr(base_module, "retrieve_timesteps", None)
    compute_empirical_mu = getattr(base_module, "compute_empirical_mu", None)
    calculate_shift = getattr(base_module, "calculate_shift", None)
    scheduler = pipeline.scheduler
    device = getattr(pipeline, "_execution_device", None)

    if callable(retrieve_timesteps) and hasattr(scheduler, "set_timesteps"):
        sigma_schedule = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if getattr(getattr(scheduler, "config", None), "use_flow_sigmas", False):
            sigma_schedule = None

        extra_kwargs = {}
        if callable(compute_empirical_mu):
            extra_kwargs["mu"] = compute_empirical_mu(image_seq_len=latents.shape[1], num_steps=num_inference_steps)
        elif callable(calculate_shift) and getattr(getattr(scheduler, "config", None), "use_dynamic_shifting", False):
            extra_kwargs["mu"] = calculate_shift(
                latents.shape[1],
                scheduler.config.get("base_image_seq_len", 256),
                scheduler.config.get("max_image_seq_len", 4096),
                scheduler.config.get("base_shift", 0.5),
                scheduler.config.get("max_shift", 1.15),
            )

        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler,
            num_inference_steps,
            device,
            sigmas=sigma_schedule,
            **extra_kwargs,
        )
    else:
        # Some schedulers do not populate `timesteps` until `set_timesteps(...)` is called.
        set_timesteps = getattr(scheduler, "set_timesteps", None)
        timesteps_attr = getattr(scheduler, "timesteps", None)
        needs_init = timesteps_attr is None or getattr(timesteps_attr, "numel", lambda: len(timesteps_attr))() == 0
        if needs_init and callable(set_timesteps):
            try:
                set_timesteps(num_inference_steps, device=device)
            except TypeError:
                set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps[:num_inference_steps]

    # Normalize to a tensor so the denoising loop can rely on `t.expand(...)` and math ops.
    if not isinstance(timesteps, torch.Tensor):
        timesteps = torch.tensor(list(timesteps), device=device)
    elif device is not None and timesteps.device != device:
        timesteps = timesteps.to(device)

    num_warmup_steps = max(len(timesteps) - num_inference_steps * getattr(scheduler, "order", 1), 0)
    return timesteps, num_inference_steps, num_warmup_steps


@contextmanager
def _restore_original_processors(pipeline):
    original_processors = getattr(pipeline, "_havedit_original_processors", None)
    if not original_processors:
        yield
        return

    current_processors = []
    for attention, original_processor in original_processors:
        current_processors.append((attention, getattr(attention, "processor", None)))
        _set_processor(attention, original_processor)

    try:
        yield
    finally:
        for attention, processor in current_processors:
            _set_processor(attention, processor)


@lru_cache(maxsize=None)
def _wrapped_pipeline_class(base_cls):
    base_module = importlib.import_module(base_cls.__module__)
    base_signature = inspect.signature(base_cls.__call__)
    output_cls = getattr(base_module, "Flux2PipelineOutput", None) or getattr(base_module, "FluxPipelineOutput", None)
    xla_available = getattr(base_module, "XLA_AVAILABLE", False)
    xm = getattr(base_module, "xm", None)

    class HAVEditPipeline(base_cls):
        @torch.no_grad()
        def __call__(self, *args, **kwargs):
            bound = base_signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            params = dict(bound.arguments)
            params.pop("self")

            image = params.get("image")
            prompt = params.get("prompt")
            negative_prompt = params.get("negative_prompt")
            # true_cfg_scale = params.get("true_cfg_scale")
            height = params.get("height")
            width = params.get("width")
            num_inference_steps = params.get("num_inference_steps")
            sigmas = params.get("sigmas")
            guidance_scale = params.get("guidance_scale")
            num_images_per_prompt = params.get("num_images_per_prompt")
            generator = params.get("generator")
            latents = params.get("latents")
            prompt_embeds = params.get("prompt_embeds")
            prompt_embeds_mask = params.get("prompt_embeds_mask")
            negative_prompt_embeds = params.get("negative_prompt_embeds")
            negative_prompt_embeds_mask = params.get("negative_prompt_embeds_mask")
            output_type = params.get("output_type")
            return_dict = params.get("return_dict")
            attention_kwargs = params.get("attention_kwargs")
            callback_on_step_end = params.get("callback_on_step_end")
            callback_on_step_end_tensor_inputs = params.get("callback_on_step_end_tensor_inputs")
            if callback_on_step_end_tensor_inputs is None:
                callback_on_step_end_tensor_inputs = list(getattr(self, "_callback_tensor_inputs", []))
            max_sequence_length = params.get("max_sequence_length")
            text_encoder_out_layers = params.get("text_encoder_out_layers")

            adapter = getattr(self, "_havedit_backend_adapter", None)
            if adapter is None:
                backend_name = getattr(self.transformer, "_havedit_backend", "flux2")
                adapter = get_backend_adapter_class(backend_name)()
                self._havedit_backend_adapter = adapter

            prepare_condition_images = getattr(adapter, "prepare_condition_images", _prepare_condition_images)
            condition_images, height, width = prepare_condition_images(self, image, height, width)
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

            adapter.check_inputs(
                self,
                SimpleNamespace(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                    guidance_scale=guidance_scale,
                    max_sequence_length=max_sequence_length,
                    condition_images=condition_images,
                ),
                callback_on_step_end_tensor_inputs,
            )

            self._guidance_scale = guidance_scale
            self._attention_kwargs = attention_kwargs
            self._current_timestep = None
            self._interrupt = False

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            run_ctx = adapter.build_run_context(
                self,
                SimpleNamespace(
                    prompt=prompt,
                    prompt_embeds=prompt_embeds,
                    prompt_embeds_mask=prompt_embeds_mask,
                    negative_prompt=negative_prompt,
                    negative_prompt_embeds=negative_prompt_embeds,
                    negative_prompt_embeds_mask=negative_prompt_embeds_mask,
                    num_images_per_prompt=num_images_per_prompt,
                    guidance_scale=guidance_scale,
                    # true_cfg_scale=true_cfg_scale,
                    max_sequence_length=max_sequence_length,
                    text_encoder_out_layers=text_encoder_out_layers,
                    generator=generator,
                    latents=latents,
                    height=height,
                    width=width,
                    condition_images=condition_images,
                    batch_size=batch_size,
                ),
            )

            prompt_embeds = run_ctx.prompt_embeds
            text_ids = run_ctx.text_ids
            negative_prompt_embeds = run_ctx.negative_prompt_embeds
            negative_text_ids = run_ctx.negative_text_ids
            latents = run_ctx.latents
            latent_ids = run_ctx.latent_ids
            image_latents = run_ctx.reference_latents
            image_latent_ids = run_ctx.reference_ids

            prepare_timesteps = getattr(adapter, "prepare_timesteps", _prepare_timesteps)
            timesteps, num_inference_steps, num_warmup_steps = prepare_timesteps(
                self,
                base_module,
                latents,
                num_inference_steps,
                sigmas,
            )
            self._num_timesteps = len(timesteps)

            state = self._havedit_state
            state.begin_run(
                total_steps=num_inference_steps,
                text_n=run_ctx.text_n,
                latent_n=run_ctx.latent_n,
                ref_n=run_ctx.ref_n,
                height=run_ctx.height,
                width=run_ctx.width,
            )
            state.latent_ids = latent_ids
            if image_latents is not None:
                initialize_source_latents(state, image_latents)

            set_begin_index = getattr(self.scheduler, "set_begin_index", None)
            if callable(set_begin_index):
                set_begin_index(0)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if getattr(self, "interrupt", getattr(self, "_interrupt", False)):
                        break

                    state.begin_step(i)
                    self._current_timestep = t
                    step_ctx = adapter.build_step_context(self, run_ctx, latents, t, guidance_scale)
                    noise_pred = adapter.run_cond_step(self, run_ctx, step_ctx)

                    state.finalize_semantic_prior_if_needed()
                    noise_pred = noise_pred[:, : latents.size(1)]

                    should_run_uncond_step = getattr(adapter, "should_run_uncond_step", None)
                    run_uncond_step = (
                        should_run_uncond_step(self, run_ctx, guidance_scale)
                        if callable(should_run_uncond_step)
                        else getattr(self, "do_classifier_free_guidance", False)
                        and run_ctx.negative_prompt_embeds is not None
                    )
                    if run_uncond_step:
                        with _restore_original_processors(self):
                            neg_noise_pred = adapter.run_uncond_step(self, run_ctx, step_ctx)
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        combine_noise_pred = getattr(adapter, "combine_noise_pred", None)
                        if callable(combine_noise_pred):
                            noise_pred = combine_noise_pred(noise_pred, neg_noise_pred, run_ctx, guidance_scale)
                        else:
                            noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                    state.latest_pred_velocity = noise_pred.detach().clone()
                    latents_dtype = latents.dtype
                    latents = adapter.scheduler_step(self, noise_pred, t, latents)
                    if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                    if callback_on_step_end is not None:
                        # Avoid relying on `locals()` in a comprehension (it has its own scope in Python 3),
                        # and ignore unknown names rather than raising KeyError.
                        callback_tensor_map = {
                            "latents": latents,
                            "latent_ids": latent_ids,
                            "prompt_embeds": prompt_embeds,
                            "negative_prompt_embeds": negative_prompt_embeds,
                            "image_latents": image_latents,
                            "image_latent_ids": image_latent_ids,
                        }
                        callback_kwargs = {
                            name: callback_tensor_map[name]
                            for name in callback_on_step_end_tensor_inputs
                            if name in callback_tensor_map
                        }
                        callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                        latents = callback_outputs.pop("latents", latents)
                        run_ctx.prompt_embeds = callback_outputs.pop("prompt_embeds", run_ctx.prompt_embeds)
                        prompt_embeds = run_ctx.prompt_embeds

                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % getattr(self.scheduler, "order", 1) == 0
                    ):
                        progress_bar.update()

                    if xla_available and xm is not None:
                        xm.mark_step()

            self._current_timestep = None
            image = adapter.decode_latents(self, latents, run_ctx, output_type, return_dict)

            maybe_free_model_hooks = getattr(self, "maybe_free_model_hooks", None)
            if callable(maybe_free_model_hooks):
                maybe_free_model_hooks()

            if not return_dict:
                return (image,)
            if output_cls is None:
                return image
            return output_cls(images=image)

    HAVEditPipeline.__name__ = f"HAVEdit{base_cls.__name__}"
    HAVEditPipeline.__qualname__ = HAVEditPipeline.__name__
    HAVEditPipeline.__module__ = base_cls.__module__
    return HAVEditPipeline


def enable_havedit(pipeline, config):
    from .attention_processor_headwise_subjectbg_subjectrelease import HAVEditFluxAttnProcessor

    if hasattr(pipeline, "_havedit_state"):
        return pipeline

    state = HAVEditRuntimeState(config=config)
    runtime_cfg = getattr(config, "runtime", config)
    backend_name = getattr(runtime_cfg, "backend", "flux2")
    pipeline.transformer._havedit_backend = backend_name
    adapter = get_backend_adapter_class(backend_name)()
    original_class = pipeline.__class__
    original_processors = []

    for block, block_name, stream_kind in adapter.iter_transformer_blocks(pipeline):
        attention = block.attn
        original_processors.append((attention, getattr(attention, "processor", None)))
        build_attention_processor = getattr(adapter, "build_attention_processor", None)
        processor = (
            build_attention_processor(
                state=state,
                block_name=block_name,
                stream_kind=stream_kind,
                default_processor_cls=HAVEditFluxAttnProcessor,
            )
            if callable(build_attention_processor)
            else HAVEditFluxAttnProcessor(
                state=state,
                block_name=block_name,
                is_single_stream=stream_kind == "single_stream",
            )
        )
        _set_processor(attention, processor)

    pipeline._havedit_state = state
    pipeline._havedit_backend_adapter = adapter
    pipeline._havedit_original_class = original_class
    pipeline._havedit_original_processors = original_processors
    pipeline.__class__ = _wrapped_pipeline_class(original_class)
    return pipeline


def disable_havedit(pipeline):
    for attention, processor in getattr(pipeline, "_havedit_original_processors", []):
        _set_processor(attention, processor)

    original_class = getattr(pipeline, "_havedit_original_class", None)
    if original_class is not None:
        pipeline.__class__ = original_class

    for name in ("_havedit_state", "_havedit_backend_adapter", "_havedit_original_class", "_havedit_original_processors"):
        if hasattr(pipeline, name):
            delattr(pipeline, name)

    return pipeline


__all__ = ["disable_havedit", "enable_havedit"]
