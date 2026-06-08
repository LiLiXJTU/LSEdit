from __future__ import annotations

import argparse
import csv
import datetime as dt
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from transformers import pipeline
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def build_generation_config(args: argparse.Namespace):
    from lsedit import LSEditConfig

    config = LSEditConfig()
    config.runtime.backend = getattr(args, "backend", config.runtime.backend)
    config.runtime.model_path = args.model_path
    config.runtime.gpu_id = args.gpu_id
    config.runtime.enable_cpu_offload = args.enable_cpu_offload
    config.bhc.enabled = getattr(args, "enable_bhc", getattr(args, "enable_bvp", True))
    config.bhc.tau_low = getattr(args, "bhc_tau_low", getattr(args, "bvp_tau_low", config.bhc.tau_low))
    config.bhc.tau_high = getattr(args, "bhc_tau_high", getattr(args, "bvp_tau_high", config.bhc.tau_high))
    config.bhc.lambda_max = getattr(
        args,
        "bhc_lambda_max",
        getattr(args, "bvp_lambda_max", config.bhc.lambda_max),
    )
    config.wsp.warmup_steps = args.warmup_steps
    config.havsr.alpha = args.alpha
    config.havsr.beta = args.beta
    config.havsr.threshold = args.threshold
    config.runtime.background_pixel_ring_width = int(getattr(args, "background_pixel_ring_width", 1))
    config.runtime.background_pixel_ring_alpha = float(getattr(args, "background_pixel_ring_alpha", 0.5))
    return config


def ensure_output_parent(output: str | Path) -> None:
    Path(output).parent.mkdir(parents=True, exist_ok=True)


def load_local_pipeline(runtime_cfg):
    from lsedit.flux import load_local_pipeline as _load_local_pipeline

    return _load_local_pipeline(runtime_cfg)


def enable_lsedit(pipeline, config):
    from lsedit import enable_lsedit as _enable_lsedit

    return _enable_lsedit(pipeline, config)


def load_input_image(image_path: str | Path):
    from PIL import Image

    with Image.open(Path(image_path)) as image:
        return image.convert("RGB")


def _call_parameters(pipeline) -> dict[str, inspect.Parameter]:
    try:
        return dict(inspect.signature(pipeline.__call__).parameters)
    except (TypeError, ValueError):
        return {}


def _accepts_kwargs(parameters: dict[str, inspect.Parameter]) -> bool:
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())


def image_argument_name(pipeline) -> str:
    parameters = _call_parameters(pipeline)
    if "image" in parameters:
        return "image"
    if "images" in parameters:
        return "images"
    return "image"


def supports_parameter(pipeline, name: str) -> bool:
    parameters = _call_parameters(pipeline)
    return name in parameters or _accepts_kwargs(parameters)


def _normalize_device_string(device_like) -> str | None:
    if device_like is None:
        return None
    if isinstance(device_like, torch.device):
        return str(device_like)
    if isinstance(device_like, str):
        return device_like
    device_type = getattr(device_like, "type", None)
    device_index = getattr(device_like, "index", None)
    if isinstance(device_type, str):
        if device_index is None:
            return device_type
        return f"{device_type}:{device_index}"
    return None


def resolve_generator_device(pipeline, args: argparse.Namespace, *, cuda_available: bool | None = None) -> str:
    for attr_name in ("_execution_device", "device"):
        device_name = _normalize_device_string(getattr(pipeline, attr_name, None))
        if device_name:
            return device_name

    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    gpu_id = getattr(args, "gpu_id", None)
    if gpu_id is not None and cuda_available:
        return f"cuda:{gpu_id}"
    return "cpu"


def build_torch_generator(pipeline, args: argparse.Namespace):
    seed = getattr(args, "seed", None)
    if seed is None:
        return None

    device = resolve_generator_device(pipeline, args)
    return torch.Generator(device=device).manual_seed(seed)


def build_pipeline_call_kwargs(pipeline, args: argparse.Namespace, input_image) -> dict[str, object]:
    call_kwargs = {
        "prompt": args.prompt,
        image_argument_name(pipeline): input_image,
        "num_inference_steps": args.num_inference_steps,
        "output_type": "pil",
        "return_dict": True,
    }
    # if supports_parameter(pipeline, "true_cfg_scale"):
    #     call_kwargs["true_cfg_scale"] = args.guidance_scale
    #     if supports_parameter(pipeline, "negative_prompt"):
    #         call_kwargs["negative_prompt"] = getattr(args, "negative_prompt", "")
    if supports_parameter(pipeline, "guidance_scale"):
        call_kwargs["guidance_scale"] = args.guidance_scale
    generator = build_torch_generator(pipeline, args)
    if generator is not None and supports_parameter(pipeline, "generator"):
        call_kwargs["generator"] = generator

    step_callback = getattr(args, "_step_visualizer_callback", None)
    if step_callback is not None and supports_parameter(pipeline, "callback_on_step_end"):
        call_kwargs["callback_on_step_end"] = step_callback

    step_callback_tensor_inputs = getattr(args, "_step_visualizer_tensor_inputs", None)
    if step_callback_tensor_inputs is not None and supports_parameter(pipeline, "callback_on_step_end_tensor_inputs"):
        call_kwargs["callback_on_step_end_tensor_inputs"] = step_callback_tensor_inputs

    return call_kwargs


def _normalize_to_uint8(array_2d: np.ndarray) -> np.ndarray:
    array_2d = np.asarray(array_2d, dtype=np.float32)
    finite = np.isfinite(array_2d)
    if not finite.any():
        return np.zeros_like(array_2d, dtype=np.uint8)

    values = array_2d[finite]
    low = float(values.min())
    high = float(values.max())
    if abs(high - low) < 1e-8:
        return np.zeros_like(array_2d, dtype=np.uint8)

    normalized = (array_2d - low) / (high - low)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _heatmap_from_2d(array_2d: np.ndarray, target_size: tuple[int, int] | None = None) -> Image.Image:
    values = _normalize_to_uint8(array_2d).astype(np.float32) / 255.0
    red = np.clip(1.5 * values - 0.5, 0.0, 1.0)
    green = np.clip(1.0 - np.abs(2.0 * values - 1.0), 0.0, 1.0)
    blue = np.clip(1.0 - 1.5 * values, 0.0, 1.0)
    rgb = np.stack([red, green, blue], axis=-1)
    heatmap = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
    if target_size is not None:
        heatmap = heatmap.resize(target_size, resample=Image.Resampling.BILINEAR)
    return heatmap


def _load_overlay_mask(path: str | Path, target_size: tuple[int, int]) -> np.ndarray:
    with Image.open(Path(path)) as image:
        mask = image.convert("L").resize(target_size, resample=Image.Resampling.NEAREST)
    values = np.asarray(mask, dtype=np.float32) / 255.0
    return np.clip(values, 0.0, 1.0)


def _overlay_mask_on_image(
    image: Image.Image,
    mask_values: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> Image.Image:
    image_rgb = image.convert("RGB")
    image_array = np.asarray(image_rgb, dtype=np.float32)
    if mask_values.shape != image_array.shape[:2]:
        raise ValueError(
            f"overlay mask shape {mask_values.shape} must match image size {image_array.shape[:2]}"
        )
    overlay_color = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blend = np.clip(mask_values.astype(np.float32), 0.0, 1.0)[..., None] * float(alpha)
    composed = image_array * (1.0 - blend) + overlay_color * blend
    return Image.fromarray(np.clip(composed, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _first_tensor_2d(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.ndim == 3:
        return tensor[0]
    if tensor.ndim == 2:
        return tensor
    return None


def _tensor_2d_to_numpy(tensor_2d: torch.Tensor) -> np.ndarray:
    return tensor_2d.detach().to(dtype=torch.float32).cpu().numpy()


def _finite_tensor_stats(tensor_2d: torch.Tensor | None) -> tuple[float, float, float]:
    if tensor_2d is None:
        return float("nan"), float("nan"), float("nan")
    values = tensor_2d.detach().to(dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(finite.mean().item()),
        float(finite.std(unbiased=False).item()),
        float(finite.max().item()),
    )


def decode_latent_image(pipeline, latents: torch.Tensor | None, latent_ids, image_size: tuple[int, int] | None = None):
    if not isinstance(latents, torch.Tensor):
        return None

    adapter = getattr(pipeline, "_lsedit_backend_adapter", None)
    if adapter is not None:
        image_width, image_height = image_size if image_size is not None else (0, 0)
        run_ctx = SimpleNamespace(
            latent_ids=latent_ids,
            image_width=image_width,
            image_height=image_height,
        )
        try:
            decoded = adapter.decode_latents(pipeline, latents.detach().clone(), run_ctx, "pil", True)
        except RuntimeError:
            decoded = None
        if isinstance(decoded, list) and decoded:
            return decoded[0]
        if decoded is not None:
            return decoded

    if image_size is not None and callable(getattr(pipeline, "_unpack_latents", None)):
        vae = getattr(pipeline, "vae", None)
        image_processor = getattr(pipeline, "image_processor", None)
        if vae is not None and image_processor is not None:
            image_width, image_height = image_size
            latent_tensor = pipeline._unpack_latents(
                latents.detach().clone(),
                image_height,
                image_width,
                pipeline.vae_scale_factor,
            )
            latent_tensor = (latent_tensor / vae.config.scaling_factor) + vae.config.shift_factor
            with torch.no_grad():
                decoded = vae.decode(latent_tensor, return_dict=False)[0]
            post = image_processor.postprocess(decoded.detach(), output_type="pil")
            if isinstance(post, list) and post:
                return post[0]
            return post

    if latent_ids is None or not callable(getattr(pipeline, "_unpack_latents_with_ids", None)):
        return None

    vae = getattr(pipeline, "vae", None)
    image_processor = getattr(pipeline, "image_processor", None)
    if vae is None or image_processor is None:
        return None
    if not callable(getattr(vae, "decode", None)) or not callable(getattr(image_processor, "postprocess", None)):
        return None
    if not callable(getattr(pipeline, "_unpatchify_latents", None)):
        return None

    latent_tensor = latents.detach().clone()
    latent_tensor = pipeline._unpack_latents_with_ids(latent_tensor, latent_ids)

    bn = getattr(vae, "bn", None)
    config = getattr(vae, "config", None)
    bn_eps = getattr(config, "batch_norm_eps", None)
    if (
        bn is not None
        and isinstance(getattr(bn, "running_mean", None), torch.Tensor)
        and isinstance(getattr(bn, "running_var", None), torch.Tensor)
        and bn_eps is not None
    ):
        bn_mean = bn.running_mean.view(1, -1, 1, 1).to(latent_tensor.device, latent_tensor.dtype)
        bn_std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + bn_eps).to(latent_tensor.device, latent_tensor.dtype)
        latent_tensor = latent_tensor * bn_std + bn_mean

    latent_tensor = pipeline._unpatchify_latents(latent_tensor)
    with torch.no_grad():
        decoded = vae.decode(latent_tensor, return_dict=False)[0]
    post = image_processor.postprocess(decoded.detach(), output_type="pil")
    if isinstance(post, list) and post:
        return post[0]
    return post


def _top_fraction_mean_ratio(tensor_2d: torch.Tensor | None, *, fraction: float) -> float:
    if tensor_2d is None:
        return float("nan")
    values = tensor_2d.detach().to(dtype=torch.float32).reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    mean = float(finite.mean().item())
    if abs(mean) < 1e-8:
        return float("nan")
    count = max(int(np.ceil(float(finite.numel()) * float(fraction))), 1)
    top_values, _ = torch.topk(finite, k=count)
    return float(top_values.mean().item() / mean)


def _area_ratio_above(tensor_2d: torch.Tensor | None, *, threshold: float) -> float:
    if tensor_2d is None:
        return float("nan")
    values = tensor_2d.detach().to(dtype=torch.float32).reshape(-1)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return float("nan")
    return float((finite >= threshold).to(dtype=torch.float32).mean().item())


def _normalize_array_2d(array_2d: np.ndarray) -> np.ndarray:
    values = np.asarray(array_2d, dtype=np.float32)
    mn = float(values.min())
    mx = float(values.max())
    return (values - mn) / (mx - mn + 1e-8)


def _component_stats(array_2d: np.ndarray, *, quantile: float) -> tuple[float, int, np.ndarray]:
    normalized = _normalize_array_2d(array_2d)
    threshold = float(np.quantile(normalized.reshape(-1), quantile))
    mask = normalized >= threshold
    labeled, num = ndimage.label(mask)
    if num == 0:
        return float("nan"), 0, mask
    sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
    sizes = np.asarray(sizes, dtype=np.float32)
    total = float(sizes.sum())
    largest = float(sizes.max()) if sizes.size else 0.0
    return largest / (total + 1e-8), int(num), mask


def _dilated_containment_ratio(
    frozen_array_2d: np.ndarray,
    current_array_2d: np.ndarray,
    *,
    quantile: float,
    dilate_iter: int,
) -> float:
    frozen = _normalize_array_2d(frozen_array_2d)
    current = _normalize_array_2d(current_array_2d)
    frozen_threshold = float(np.quantile(frozen.reshape(-1), quantile))
    current_threshold = float(np.quantile(current.reshape(-1), quantile))
    frozen_mask = frozen >= frozen_threshold
    current_mask = current >= current_threshold
    frozen_dilated = ndimage.binary_dilation(frozen_mask, iterations=dilate_iter)
    covered = np.logical_and(current_mask, frozen_dilated).sum()
    total = current_mask.sum()
    return float(covered / (total + 1e-8)) if total else float("nan")


class _StepVisualizationExporter:
    def __init__(
        self,
        *,
        output_root: Path,
        output_size: tuple[int, int],
        every_n: int,
        save_step_image: bool,
        save_step_semantic_prior: bool,
        save_step_attention: bool,
        save_step_edit_mask: bool,
        save_step_preserve_weight: bool,
        save_step_source_latent_decode: bool = False,
        save_step_metrics: bool,
        overlay_mask_path: str | Path | None = None,
        overlay_mask_alpha: float = 0.35,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.output_size = output_size
        self.every_n = max(int(every_n), 1)
        self.save_step_image = save_step_image
        self.save_step_semantic_prior = save_step_semantic_prior
        self.save_step_attention = save_step_attention
        self.save_step_edit_mask = save_step_edit_mask
        self.save_step_preserve_weight = save_step_preserve_weight
        self.save_step_source_latent_decode = save_step_source_latent_decode
        self.save_step_metrics = save_step_metrics
        self.overlay_mask_alpha = float(overlay_mask_alpha)
        self.overlay_mask_values = None
        if overlay_mask_path:
            self.overlay_mask_values = _load_overlay_mask(overlay_mask_path, self.output_size)
        self.saved_counts = {
            "image": 0,
            "semantic_prior": 0,
            "semantic_prior_current": 0,
            "attention": 0,
            "edit_mask": 0,
            "confusion_mask": 0,
            "background_mask": 0,
            "preserve_weight": 0,
            "dev_term": 0,
            "passive_mask": 0,
            "passive_keep_map": 0,
            "source_latent_decode": 0,
            "source_latent_decode_diff_heatmap": 0,
        }
        self.skipped_counts = {
            "semantic_prior": 0,
            "semantic_prior_current": 0,
            "attention": 0,
            "edit_mask": 0,
            "confusion_mask": 0,
            "background_mask": 0,
            "preserve_weight": 0,
            "dev_term": 0,
            "passive_mask": 0,
            "passive_keep_map": 0,
            "source_latent_decode": 0,
            "source_latent_decode_diff_heatmap": 0,
        }
        self.error_count = 0
        self.metric_rows: list[dict[str, float | int]] = []
        self.errors: list[dict[str, object]] = []

    def callback(self, pipeline, step_index: int, _timestep, callback_kwargs: dict[str, object]) -> dict[str, object]:
        if step_index % self.every_n != 0:
            return {}

        step_number = step_index + 1
        state = getattr(pipeline, "_lsedit_state", None)

        if self.save_step_image:
            self._run_step_export(step_number, "image", lambda: self._save_step_image(step_number, pipeline, callback_kwargs))
        if self.save_step_semantic_prior:
            self._run_step_export(step_number, "semantic_prior", lambda: self._save_semantic_prior(step_number, state))
            self._run_step_export(step_number, "semantic_prior_current", lambda: self._save_semantic_prior_current(step_number, state))
        if self.save_step_attention:
            self._run_step_export(step_number, "attention", lambda: self._save_attention(step_number, state))
        if self.save_step_edit_mask:
            self._run_step_export(step_number, "edit_mask", lambda: self._save_edit_mask(step_number, state))
            self._run_step_export(step_number, "confusion_mask", lambda: self._save_confusion_mask(step_number, state))
            self._run_step_export(step_number, "background_mask", lambda: self._save_background_mask(step_number, state))
            self._run_step_export(step_number, "passive_mask", lambda: self._save_passive_mask(step_number, state))
            self._run_step_export(step_number, "passive_keep_map", lambda: self._save_passive_keep_map(step_number, state))
        if self.save_step_preserve_weight:
            self._run_step_export(step_number, "preserve_weight", lambda: self._save_preserve_weight(step_number, state))
            self._run_step_export(step_number, "dev_term", lambda: self._save_dev_term(step_number, state))
        if self.save_step_source_latent_decode:
            self._run_step_export(
                step_number,
                "source_latent_decode",
                lambda: self._save_source_latent_decode(step_number, pipeline, state),
            )
        if self.save_step_metrics:
            self._run_step_export(step_number, "metrics", lambda: self.metric_rows.append(self._build_metric_row(step_number, state)))

        return {}

    def _run_step_export(self, step_number: int, category: str, fn) -> None:
        try:
            fn()
        except Exception as exc:
            self.error_count += 1
            self.errors.append(
                {
                    "step": int(step_number),
                    "category": category,
                    "error": repr(exc),
                }
            )

    def _save_step_image(self, step_number: int, pipeline, callback_kwargs: dict[str, object]) -> None:
        image = self._decode_step_image(pipeline, callback_kwargs)
        if image is None:
            return
        if self.overlay_mask_values is not None:
            image = _overlay_mask_on_image(
                image,
                self.overlay_mask_values,
                alpha=self.overlay_mask_alpha,
            )
        image.save(self.output_root / f"step_{step_number:04d}_image.png")
        self.saved_counts["image"] += 1

    def _save_semantic_prior(self, step_number: int, state) -> None:
        semantic_prior = getattr(state, "semantic_prior", None) if state is not None else None
        prior_map = _first_tensor_2d(semantic_prior)
        if prior_map is None:
            self.skipped_counts["semantic_prior"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(prior_map), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_semantic_prior.png")
        self.saved_counts["semantic_prior"] += 1

    def _save_semantic_prior_current(self, step_number: int, state) -> None:
        semantic_prior_current = getattr(state, "latest_semantic_prior_current", None) if state is not None else None
        prior_map = _first_tensor_2d(semantic_prior_current)
        if prior_map is None:
            self.skipped_counts["semantic_prior_current"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(prior_map), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_semantic_prior_current.png")
        self.saved_counts["semantic_prior_current"] += 1

    def _save_attention(self, step_number: int, state) -> None:
        attention_map = getattr(state, "latest_attention_map", None) if state is not None else None
        attention_2d = _first_tensor_2d(attention_map)
        if attention_2d is None:
            self.skipped_counts["attention"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(attention_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_attention.png")
        self.saved_counts["attention"] += 1

    def _save_edit_mask(self, step_number: int, state) -> None:
        edit_mask = getattr(state, "latest_edit_mask", None) if state is not None else None
        edit_mask_2d = _first_tensor_2d(edit_mask)
        if edit_mask_2d is None:
            self.skipped_counts["edit_mask"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(edit_mask_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_edit_mask.png")
        self.saved_counts["edit_mask"] += 1

    def _save_confusion_mask(self, step_number: int, state) -> None:
        confusion_mask = getattr(state, "latest_confusion_mask", None) if state is not None else None
        confusion_mask_2d = _first_tensor_2d(confusion_mask)
        if confusion_mask_2d is None:
            self.skipped_counts["confusion_mask"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(confusion_mask_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_confusion_mask.png")
        self.saved_counts["confusion_mask"] += 1

    def _save_background_mask(self, step_number: int, state) -> None:
        background_mask = getattr(state, "latest_background_mask", None) if state is not None else None
        background_mask_2d = _first_tensor_2d(background_mask)
        if background_mask_2d is None:
            self.skipped_counts["background_mask"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(background_mask_2d), target_size=self.output_size)
        if self.overlay_mask_values is not None:
            heatmap = _overlay_mask_on_image(
                heatmap,
                self.overlay_mask_values,
                alpha=self.overlay_mask_alpha,
            )
        heatmap.save(self.output_root / f"step_{step_number:04d}_background_mask.png")
        self.saved_counts["background_mask"] += 1

    def _save_passive_mask(self, step_number: int, state) -> None:
        passive_mask = getattr(state, "latest_passive_mask", None) if state is not None else None
        passive_mask_2d = _first_tensor_2d(passive_mask)
        if passive_mask_2d is None:
            self.skipped_counts["passive_mask"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(passive_mask_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_passive_mask.png")
        self.saved_counts["passive_mask"] += 1

    def _save_passive_keep_map(self, step_number: int, state) -> None:
        passive_keep_map = getattr(state, "latest_passive_keep_map", None) if state is not None else None
        passive_keep_2d = _first_tensor_2d(passive_keep_map)
        if passive_keep_2d is None:
            self.skipped_counts["passive_keep_map"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(passive_keep_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_passive_keep_map.png")
        self.saved_counts["passive_keep_map"] += 1

    def _save_preserve_weight(self, step_number: int, state) -> None:
        preserve_weight_map = getattr(state, "latest_preserve_weight_map", None) if state is not None else None
        preserve_weight_2d = _first_tensor_2d(preserve_weight_map)
        if preserve_weight_2d is None:
            self.skipped_counts["preserve_weight"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(preserve_weight_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_preserve_weight.png")
        self.saved_counts["preserve_weight"] += 1

    def _save_dev_term(self, step_number: int, state) -> None:
        dev_term_map = getattr(state, "latest_dev_term_map", None) if state is not None else None
        dev_term_2d = _first_tensor_2d(dev_term_map)
        if dev_term_2d is None:
            self.skipped_counts["dev_term"] += 1
            return
        heatmap = _heatmap_from_2d(_tensor_2d_to_numpy(dev_term_2d), target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_dev_term.png")
        self.saved_counts["dev_term"] += 1

    def _save_source_latent_decode(self, step_number: int, pipeline, state) -> None:
        source_latents = getattr(state, "source_latents", None) if state is not None else None
        latent_ids = getattr(state, "latent_ids", None) if state is not None else None
        source_image = getattr(state, "source_image", None) if state is not None else None
        image_size = source_image.size if isinstance(source_image, Image.Image) else None
        image = decode_latent_image(pipeline, source_latents, latent_ids, image_size=image_size)
        if image is None:
            self.skipped_counts["source_latent_decode"] += 1
            self.skipped_counts["source_latent_decode_diff_heatmap"] += 1
            return
        image = image.convert("RGB").resize(self.output_size, resample=Image.Resampling.BILINEAR)
        image.save(self.output_root / f"step_{step_number:04d}_source_latent_decode.png")
        self.saved_counts["source_latent_decode"] += 1

        if not isinstance(source_image, Image.Image):
            self.skipped_counts["source_latent_decode_diff_heatmap"] += 1
            return
        source_rgb = np.asarray(source_image.convert("RGB").resize(self.output_size, resample=Image.Resampling.BILINEAR))
        decoded_rgb = np.asarray(image)
        diff = np.abs(decoded_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=-1)
        heatmap = _heatmap_from_2d(diff, target_size=self.output_size)
        heatmap.save(self.output_root / f"step_{step_number:04d}_source_latent_decode_diff_heatmap.png")
        self.saved_counts["source_latent_decode_diff_heatmap"] += 1

    def _build_metric_row(self, step_number: int, state) -> dict[str, float | int]:
        semantic_prior = getattr(state, "semantic_prior", None) if state is not None else None
        semantic_prior_2d = _first_tensor_2d(semantic_prior)
        semantic_prior_current = getattr(state, "latest_semantic_prior_current", None) if state is not None else None
        semantic_prior_current_2d = _first_tensor_2d(semantic_prior_current)
        semantic_prior_mean, semantic_prior_std, semantic_prior_max = _finite_tensor_stats(semantic_prior_2d)
        preserve_weight_map = getattr(state, "latest_preserve_weight_map", None) if state is not None else None
        preserve_weight_2d = _first_tensor_2d(preserve_weight_map)
        preserve_mean_map, _preserve_std_map, preserve_max = _finite_tensor_stats(preserve_weight_2d)
        preserve_min = float("nan")
        if preserve_weight_2d is not None:
            preserve_values = preserve_weight_2d.detach().to(dtype=torch.float32)
            finite = preserve_values[torch.isfinite(preserve_values)]
            if finite.numel() > 0:
                preserve_min = float(finite.min().item())

        edit_mask = getattr(state, "latest_edit_mask", None) if state is not None else None
        edit_mask_2d = _first_tensor_2d(edit_mask)
        edit_ratio = float("nan")
        if edit_mask_2d is not None:
            edit_ratio = float(edit_mask_2d.detach().to(dtype=torch.float32).mean().item())

        def _as_float(name: str) -> float:
            value = getattr(state, name, None) if state is not None else None
            if value is None:
                return float("nan")
            return float(value)

        largest_component_ratio = float("nan")
        component_count = 0
        containment_ratio = float("nan")
        if semantic_prior_2d is not None and semantic_prior_current_2d is not None:
            current_array = _tensor_2d_to_numpy(semantic_prior_current_2d)
            frozen_array = _tensor_2d_to_numpy(semantic_prior_2d)
            largest_component_ratio, component_count, _ = _component_stats(current_array, quantile=0.9)
            containment_ratio = _dilated_containment_ratio(
                frozen_array,
                current_array,
                quantile=0.9,
                dilate_iter=3,
            )

        return {
            "step": int(step_number),
            "semantic_prior_mean": semantic_prior_mean,
            "semantic_prior_std": semantic_prior_std,
            "semantic_prior_max": semantic_prior_max,
            "semantic_prior_top10_mean_ratio": _top_fraction_mean_ratio(semantic_prior_2d, fraction=0.10),
            "semantic_prior_current_vs_frozen_largest_component_ratio": largest_component_ratio,
            "semantic_prior_current_vs_frozen_component_count": int(component_count),
            "semantic_prior_current_in_frozen_dilated_ratio": containment_ratio,
            "preserve_score_mean": _as_float("latest_preserve_score_mean"),
            "preserve_weight_mean": _as_float("latest_preserve_weight_mean"),
            "preserve_weight_min": preserve_min,
            "preserve_weight_max": preserve_max,
            "preserve_weight_max_over_mean": (
                float(preserve_max / preserve_mean_map)
                if np.isfinite(preserve_max) and np.isfinite(preserve_mean_map) and abs(preserve_mean_map) >= 1e-8
                else float("nan")
            ),
            "high_preserve_area_ratio": _area_ratio_above(preserve_weight_2d, threshold=0.8),
            "edit_ratio": edit_ratio,
            "attention_js_ref_vs_pred": _as_float("latest_attention_js_ref_vs_pred"),
            "consistency_score_mean": _as_float("latest_consistency_score_mean"),
            "release_score_mean": _as_float("latest_release_score_mean"),
            "trust_score_mean": _as_float("latest_trust_score_mean"),
        }

    def _decode_step_image(self, pipeline, callback_kwargs: dict[str, object]):
        latents = callback_kwargs.get("latents")
        latent_ids = callback_kwargs.get("latent_ids")
        image_size = None
        if latent_ids is None:
            state = getattr(pipeline, "_lsedit_state", None)
            latent_ids = getattr(state, "latent_ids", None) if state is not None else None
        state = getattr(pipeline, "_lsedit_state", None)
        source_image = getattr(state, "source_image", None) if state is not None else None
        if isinstance(source_image, Image.Image):
            image_size = source_image.size
        return decode_latent_image(pipeline, latents, latent_ids, image_size=image_size)

    def summary_text(self) -> str:
        return (
            f"Step visualization output: {self.output_root} | "
            f"saved(image={self.saved_counts['image']}, semantic_prior={self.saved_counts['semantic_prior']}, "
            f"semantic_prior_current={self.saved_counts['semantic_prior_current']}, "
            f"attention={self.saved_counts['attention']}, edit_mask={self.saved_counts['edit_mask']}, "
            f"preserve_weight={self.saved_counts['preserve_weight']}) | "
            f"skipped(semantic_prior={self.skipped_counts['semantic_prior']}, "
            f"semantic_prior_current={self.skipped_counts['semantic_prior_current']}, "
            f"attention={self.skipped_counts['attention']}, edit_mask={self.skipped_counts['edit_mask']}, "
            f"preserve_weight={self.skipped_counts['preserve_weight']}) | "
            f"metrics={len(self.metric_rows)} | errors={self.error_count}"
        )

    def finalize(self) -> None:
        if self.save_step_metrics:
            metrics_json_path = self.output_root / "metrics.json"
            metrics_csv_path = self.output_root / "metrics.csv"
            payload = {"rows": self.metric_rows}
            metrics_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            fieldnames = [
                "step",
                "semantic_prior_mean",
                "semantic_prior_std",
                "semantic_prior_max",
                "semantic_prior_top10_mean_ratio",
                "semantic_prior_current_vs_frozen_largest_component_ratio",
                "semantic_prior_current_vs_frozen_component_count",
                "semantic_prior_current_in_frozen_dilated_ratio",
                "preserve_score_mean",
                "preserve_weight_mean",
                "preserve_weight_min",
                "preserve_weight_max",
                "preserve_weight_max_over_mean",
                "high_preserve_area_ratio",
                "edit_ratio",
                "attention_js_ref_vs_pred",
                "consistency_score_mean",
                "release_score_mean",
                "trust_score_mean",
            ]
            with metrics_csv_path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.metric_rows:
                    writer.writerow(row)
        if self.errors:
            (self.output_root / "errors.json").write_text(json.dumps({"errors": self.errors}, ensure_ascii=False, indent=2))


def _build_step_visualizer(args: argparse.Namespace, output_path: Path, source_image) -> _StepVisualizationExporter | None:
    if not bool(getattr(args, "visualize_steps", False)):
        return None

    visualize_dir = Path(
        getattr(args, "visualize_dir", "/data/ll/output/LSEdit_PIE_Bench_show")
    )
    run_id = dt.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = visualize_dir / run_id / Path(output_path).stem
    output_size = getattr(source_image, "size", None)
    if not (isinstance(output_size, tuple) and len(output_size) == 2):
        output_size = (512, 512)

    return _StepVisualizationExporter(
        output_root=run_dir,
        output_size=(int(output_size[0]), int(output_size[1])),
        every_n=int(getattr(args, "visualize_every_n", 1)),
        save_step_image=bool(getattr(args, "save_step_image", True)),
        save_step_semantic_prior=bool(getattr(args, "save_step_semantic_prior", True)),
        save_step_attention=bool(getattr(args, "save_step_attention", True)),
        save_step_edit_mask=bool(getattr(args, "save_step_edit_mask", True)),
        save_step_preserve_weight=bool(getattr(args, "save_step_preserve_weight", True)),
        save_step_source_latent_decode=bool(getattr(args, "save_step_source_latent_decode", False)),
        save_step_metrics=bool(getattr(args, "save_step_metrics", True)),
        overlay_mask_path=getattr(args, "overlay_mask_path", None),
        overlay_mask_alpha=float(getattr(args, "overlay_mask_alpha", 0.35)),
    )


def merge_step_callbacks(
    callbacks: list[Callable[..., dict[str, object]]] | None,
) -> Callable[..., dict[str, object]] | None:
    if not callbacks:
        return None
    filtered = [callback for callback in callbacks if callback is not None]
    if not filtered:
        return None
    if len(filtered) == 1:
        return filtered[0]

    def _merged_callback(pipeline, step_index: int, timestep, callback_kwargs: dict[str, object]) -> dict[str, object]:
        merged_outputs: dict[str, object] = {}
        mutable_kwargs = dict(callback_kwargs)
        for callback in filtered:
            callback_outputs = callback(pipeline, step_index, timestep, mutable_kwargs)
            if not isinstance(callback_outputs, dict):
                continue
            for key in ("latents", "prompt_embeds"):
                if key in callback_outputs:
                    mutable_kwargs[key] = callback_outputs[key]
            merged_outputs.update(callback_outputs)
        return merged_outputs

    return _merged_callback


def extract_output_image(result):
    images = getattr(result, "images", result)
    if isinstance(images, tuple):
        images = images[0]
    if isinstance(images, list):
        return images[0]
    return images


def save_prediction_image(prediction_image, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    ensure_output_parent(output_path)
    prediction_image.save(output_path)
    return output_path


def normalize_output_image_size(prediction_image, source_image):
    if not isinstance(prediction_image, Image.Image):
        return prediction_image
    if not isinstance(source_image, Image.Image):
        return prediction_image
    if prediction_image.size == source_image.size:
        return prediction_image
    return prediction_image.resize(source_image.size, resample=Image.Resampling.LANCZOS)


def maybe_paste_back_background_pixels(prediction_image, source_image, state, enabled: bool):
    if not enabled or state is None:
        return prediction_image

    background_mask = getattr(state, "frozen_background_mask", None)
    if background_mask is None:
        return prediction_image

    background_mask_2d = _first_tensor_2d(background_mask)
    if background_mask_2d is None:
        return prediction_image

    prediction_rgb = prediction_image.convert("RGB")
    source_rgb = source_image.convert("RGB")
    if source_rgb.size != prediction_rgb.size:
        source_rgb = source_rgb.resize(prediction_rgb.size, resample=Image.Resampling.BILINEAR)

    mask_image = Image.fromarray(
        (background_mask_2d.detach().to(dtype=torch.float32).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8),
        mode="L",
    ).resize(prediction_rgb.size, resample=Image.Resampling.NEAREST)
    mask = np.asarray(mask_image, dtype=np.uint8) > 0

    prediction_array = np.asarray(prediction_rgb, dtype=np.uint8).copy()
    source_array = np.asarray(source_rgb, dtype=np.uint8)
    runtime_cfg = getattr(getattr(state, "config", None), "runtime", None)
    ring_width = max(int(getattr(runtime_cfg, "background_pixel_ring_width", 1)), 0)
    ring_alpha = float(np.clip(getattr(runtime_cfg, "background_pixel_ring_alpha", 0.5), 0.0, 1.0))

    if ring_width > 0:
        subject = ~mask
        subject_dilated = ndimage.binary_dilation(subject, iterations=ring_width, border_value=0)
        ring = mask & subject_dilated
        solid_background = mask & (~ring)
    else:
        ring = np.zeros_like(mask, dtype=bool)
        solid_background = mask

    prediction_array[solid_background] = source_array[solid_background]
    if ring.any():
        blended = (
            source_array[ring].astype(np.float32) * ring_alpha
            + prediction_array[ring].astype(np.float32) * (1.0 - ring_alpha)
        )
        prediction_array[ring] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(prediction_array, mode="RGB")


def _remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    cleaned = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area >= min_area:
            cleaned |= labels == component_id
    return cleaned


def build_background_mask_from_absdiff(source_image, prediction_image, args: argparse.Namespace) -> np.ndarray:
    source_rgb = np.asarray(source_image.convert("RGB"), dtype=np.uint8)
    prediction_rgb = np.asarray(prediction_image.convert("RGB"), dtype=np.uint8)
    if source_rgb.shape != prediction_rgb.shape:
        source_rgb = np.asarray(
            source_image.convert("RGB").resize(
                (prediction_rgb.shape[1], prediction_rgb.shape[0]),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )

    diff = cv2.absdiff(source_rgb, prediction_rgb)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    threshold = int(getattr(args, "diff_mask_threshold", 16))
    edit_mask = diff_gray >= threshold

    open_kernel = max(int(getattr(args, "diff_mask_open_kernel", 3)), 1)
    close_kernel = max(int(getattr(args, "diff_mask_close_kernel", 3)), 1)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        edit_mask = cv2.morphologyEx(edit_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        edit_mask = cv2.morphologyEx(edit_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0

    min_area = max(int(getattr(args, "diff_mask_min_area", 64)), 1)
    edit_mask = _remove_small_components(edit_mask, min_area=min_area)
    return ~edit_mask


def save_diff_background_mask_visualization(background_mask: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    mask_path = output_path.with_name(f"{output_path.stem}_diff_background_mask.png")
    Image.fromarray((background_mask.astype(np.uint8) * 255), mode="L").save(mask_path)
    return mask_path


def save_source_latent_decode_visualization(pipeline, output_path: str | Path) -> Path | None:
    state = getattr(pipeline, "_lsedit_state", None)
    if state is None:
        return None
    source_image = getattr(state, "source_image", None)
    image_size = source_image.size if isinstance(source_image, Image.Image) else None
    image = decode_latent_image(
        pipeline,
        getattr(state, "source_latents", None),
        getattr(state, "latent_ids", None),
        image_size=image_size,
    )
    if image is None:
        return None
    output_path = Path(output_path)
    image_path = output_path.with_name(f"{output_path.stem}_source_latent_decode.png")
    image.save(image_path)
    if isinstance(source_image, Image.Image):
        source_rgb = np.asarray(source_image.convert("RGB"))
        decoded_rgb = np.asarray(image.convert("RGB").resize(source_image.size, resample=Image.Resampling.BILINEAR))
        diff = np.abs(decoded_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=-1)
        heatmap = _heatmap_from_2d(diff, target_size=source_image.size)
        heatmap_path = output_path.with_name(f"{output_path.stem}_source_latent_decode_diff_heatmap.png")
        heatmap.save(heatmap_path)
    return image_path


def save_source_vae_recon_visualization(pipeline, source_image: Image.Image, output_path: str | Path) -> Path | None:
    vae = getattr(pipeline, "vae", None)
    image_processor = getattr(pipeline, "image_processor", None)
    if vae is None or image_processor is None:
        return None
    if not callable(getattr(vae, "encode", None)) or not callable(getattr(vae, "decode", None)):
        return None
    if not callable(getattr(image_processor, "preprocess", None)) or not callable(getattr(image_processor, "postprocess", None)):
        return None

    width, height = source_image.size
    multiple_of = int(getattr(pipeline, "vae_scale_factor", 1)) * 2
    if multiple_of > 1:
        width = max((width // multiple_of) * multiple_of, multiple_of)
        height = max((height // multiple_of) * multiple_of, multiple_of)

    image_tensor = image_processor.preprocess(source_image, height=height, width=width, resize_mode="crop")
    vae_dtype = getattr(vae, "dtype", None)
    if isinstance(image_tensor, torch.Tensor):
        first_param = next(vae.parameters(), None)
        vae_device = first_param.device if first_param is not None else image_tensor.device
        if vae_dtype is not None:
            image_tensor = image_tensor.to(device=vae_device, dtype=vae_dtype)
        else:
            image_tensor = image_tensor.to(device=vae_device)
    with torch.no_grad():
        encoded = vae.encode(image_tensor)
        latent_dist = getattr(encoded, "latent_dist", None)
        if latent_dist is not None and callable(getattr(latent_dist, "mode", None)):
            latents = latent_dist.mode()
        elif isinstance(encoded, (tuple, list)) and encoded:
            latents = encoded[0]
        else:
            latents = encoded
        decoded = vae.decode(latents, return_dict=False)[0]
    post = image_processor.postprocess(decoded.detach(), output_type="pil")
    image = post[0] if isinstance(post, list) and post else post
    if image is None:
        return None

    output_path = Path(output_path)
    image = image.convert("RGB")
    image_path = output_path.with_name(f"{output_path.stem}_source_vae_recon.png")
    image.save(image_path)
    decoded_rgb = np.asarray(image.resize(source_image.size, resample=Image.Resampling.BILINEAR))
    source_rgb = np.asarray(source_image.convert("RGB"))
    diff = np.abs(decoded_rgb.astype(np.float32) - source_rgb.astype(np.float32)).mean(axis=-1)
    heatmap = _heatmap_from_2d(diff, target_size=source_image.size)
    heatmap_path = output_path.with_name(f"{output_path.stem}_source_vae_recon_diff_heatmap.png")
    heatmap.save(heatmap_path)
    return image_path


def maybe_paste_back_background_pixels_from_diff(prediction_image, source_image, args: argparse.Namespace):
    if not bool(getattr(args, "enable_diff_background_pasteback", False)):
        return prediction_image, None

    background_mask = build_background_mask_from_absdiff(source_image, prediction_image, args)
    prediction_rgb = prediction_image.convert("RGB")
    source_rgb = source_image.convert("RGB")
    if source_rgb.size != prediction_rgb.size:
        source_rgb = source_rgb.resize(prediction_rgb.size, resample=Image.Resampling.BILINEAR)

    prediction_array = np.asarray(prediction_rgb, dtype=np.uint8).copy()
    source_array = np.asarray(source_rgb, dtype=np.uint8)
    ring_width = max(int(getattr(args, "background_pixel_ring_width", 1)), 0)
    ring_alpha = float(np.clip(getattr(args, "background_pixel_ring_alpha", 0.5), 0.0, 1.0))

    if ring_width > 0:
        subject = ~background_mask
        subject_dilated = ndimage.binary_dilation(subject, iterations=ring_width, border_value=0)
        ring = background_mask & subject_dilated
        solid_background = background_mask & (~ring)
    else:
        ring = np.zeros_like(background_mask, dtype=bool)
        solid_background = background_mask

    prediction_array[solid_background] = source_array[solid_background]
    if ring.any():
        blended = (
            source_array[ring].astype(np.float32) * ring_alpha
            + prediction_array[ring].astype(np.float32) * (1.0 - ring_alpha)
        )
        prediction_array[ring] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(prediction_array, mode="RGB"), background_mask


def maybe_save_source_image(source_image_path: str | Path, destination_path: str | Path, save_source: bool) -> None:
    if not save_source:
        return

    source_image = load_input_image(source_image_path)
    save_prediction_image(source_image, destination_path)


def should_generate_prediction(*, prediction_exists: bool, skip_inference: bool, overwrite: bool) -> bool:
    if skip_inference:
        return False
    if overwrite:
        return True
    return not prediction_exists


def _enable_lsedit_with_optional_prompt(enable_lsedit_fn, pipeline, config, *, prompt: str):
    """Enable either the stable or conflict-aware LSEdit variant.

    The stable helper is typically `enable_lsedit(pipeline, config)`.
    The conflict-aware helper is typically `enable_conflict_aware_lsedit(pipeline, config, prompt=...)`.
    """
    try:
        signature = inspect.signature(enable_lsedit_fn)
    except (TypeError, ValueError):
        return enable_lsedit_fn(pipeline, config)

    parameters = dict(signature.parameters)
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if accepts_kwargs:
        # Some conflict-aware hooks accept prompt via **kwargs without declaring it explicitly.
        return enable_lsedit_fn(pipeline, config, prompt=prompt)

    prompt_param = parameters.get("prompt")
    if prompt_param is None:
        # A few integrations may accept prompt positionally (or via *args) without naming it `prompt`.
        accepts_varargs = any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values())
        if accepts_varargs:
            return enable_lsedit_fn(pipeline, config, prompt)

        positional_params = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional_params) >= 3:
            return enable_lsedit_fn(pipeline, config, prompt)

        return enable_lsedit_fn(pipeline, config)

    if prompt_param.kind is inspect.Parameter.POSITIONAL_ONLY:
        return enable_lsedit_fn(pipeline, config, prompt)

    return enable_lsedit_fn(pipeline, config, prompt=prompt)

def _calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def _retrieve_timesteps(scheduler, num_inference_steps: int, device, sigmas, **kwargs):
    import inspect

    if sigmas is not None:
        parameters = set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if "sigmas" not in parameters:
            raise ValueError(f"Scheduler {scheduler.__class__.__name__} does not support custom sigmas.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        return scheduler.timesteps, len(scheduler.timesteps)

    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps

def _spatial_shape_from_ids(token_ids, token_count: int) -> tuple[int, int]:
    if token_ids is None or getattr(token_ids, "numel", lambda: 0)() == 0 or token_ids.shape[-1] < 3:
        return 1, token_count
    height = int(token_ids[..., 1].max().item()) + 1
    width = int(token_ids[..., 2].max().item()) + 1
    return height, width

def _run_single_edit_flux1_kontext_with_image(
    pipe,
    *,
    image,
    num_inference_steps,
    output_type,
    return_dict,
    guidance_scale,
    generator,
    prompt: str,
    output_path: Path,
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    _auto_resize: bool = True,
    args: argparse.Namespace,
):
    import numpy as np
    import torch
    from lsedit.flux.pipeline import initialize_source_latents

    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_generator_device(pipe, args)
    source_image = image
    state = getattr(pipe, "_lsedit_state", None)
    if state is not None and hasattr(source_image, "copy"):
        state.source_image = source_image.copy()
    # gen_height, gen_width = _resolve_generation_size(
    #     pipe,
    #     image,
    #     height=args.height,
    #     width=args.width,
    #     auto_resize=not args.disable_auto_resize,
    # )
    multiple_of = pipe.vae_scale_factor * 2
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == pipe.latent_channels):
            img = image[0] if isinstance(image, list) else image
            image_height, image_width = pipe.image_processor.get_default_height_width(img)
            aspect_ratio = image_width / image_height
            if _auto_resize:
                # Kontext is trained on specific resolutions, using one of them is recommended
                _, image_width, image_height = min(
                    (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = pipe.image_processor.resize(image, image_height, image_width)
            image = pipe.image_processor.preprocess(image, image_height, image_width)
            height, width = image.shape[-2], image.shape[-1]

    else:
        height = height or pipe.default_sample_size * pipe.vae_scale_factor
        width = width  or pipe.default_sample_size * pipe.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
    # condition_image = _preprocess_condition_image(pipe, image, height=gen_height, width=gen_width)
    generator = build_torch_generator(pipe, args)

    pipe.check_inputs(
        prompt=prompt,
        prompt_2=None,
        height=height,
        width=width,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=max_sequence_length,
    )

    prompt_embeds, pooled_prompt_embeds, text_ids = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = pipe.prepare_latents(
        image,
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )
    denoise_latent_ids = latent_ids
    if image_ids is not None:
        denoise_latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension
    sigmas = np.linspace(1.0, 1 / args.num_inference_steps, args.num_inference_steps)
    if getattr(getattr(pipe.scheduler, "config", None), "use_flow_sigmas", False):
        sigmas = None

    mu = _calculate_shift(
        latents.shape[1],
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, total_steps = _retrieve_timesteps(
        pipe.scheduler,
        args.num_inference_steps,
        device,
        sigmas,
        mu=mu,
    )

    guidance = None
    if getattr(pipe.transformer.config, "guidance_embeds", False):
        guidance = torch.full([1], args.guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])

    latent_height, latent_width = _spatial_shape_from_ids(latent_ids, latents.shape[1])
    state = getattr(pipe, "_lsedit_state", None)
    if state is not None:
        state.begin_run(
            total_steps=total_steps,
            text_n=prompt_embeds.shape[1],
            latent_n=latents.shape[1],
            ref_n=0 if image_latents is None else image_latents.shape[1],
            height=latent_height,
            width=latent_width,
        )
        state.latent_ids = latent_ids
        if hasattr(source_image, "copy"):
            state.source_image = source_image.copy()
        if image_latents is not None:
            initialize_source_latents(state, image_latents)
    step_visualizer = _build_step_visualizer(args, output_path, source_image)

    set_begin_index = getattr(pipe.scheduler, "set_begin_index", None)
    if callable(set_begin_index):
        set_begin_index(0)

    with torch.inference_mode():
        progress_bar_context = pipe.progress_bar(total=total_steps) if hasattr(pipe, "progress_bar") else None
        if progress_bar_context is None:
            progress_bar = None
            progress_manager = None
        else:
            progress_manager = progress_bar_context
            progress_bar = progress_manager.__enter__()
        try:
            for step_index, timestep in enumerate(timesteps):
                expanded_timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
                if state is not None:
                    state.begin_step(step_index)
                    if image_latents is not None and hasattr(state, "begin_reference_value_capture"):
                        state.begin_reference_value_capture()
                        clean_timestep = torch.zeros_like(expanded_timestep)
                        try:
                            pipe.transformer(
                                hidden_states=image_latents,
                                timestep=clean_timestep / 1000,
                                guidance=guidance,
                                pooled_projections=pooled_prompt_embeds,
                                encoder_hidden_states=prompt_embeds,
                                txt_ids=text_ids,
                                img_ids=image_ids,
                                joint_attention_kwargs=None,
                                return_dict=False,
                            )
                        finally:
                            state.end_reference_value_capture()
                latent_model_input = latents
                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1)
                noise_pred = pipe.transformer(
                    hidden_states=latent_model_input,
                    timestep=expanded_timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=denoise_latent_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0]
                if state is not None:
                    state.finalize_semantic_prior_if_needed()
                    state.latest_pred_velocity = noise_pred[:, : latents.size(1)].detach().clone()
                latents = pipe.scheduler.step(noise_pred[:, : latents.size(1)], timestep, latents, return_dict=False)[0]
                if step_visualizer is not None:
                    step_visualizer.callback(
                        pipe,
                        step_index,
                        timestep,
                        {"latents": latents, "latent_ids": latent_ids},
                    )
                if progress_bar is not None:
                    progress_bar.update()
        finally:
            if progress_manager is not None:
                progress_manager.__exit__(None, None, None)
    if step_visualizer is not None:
        step_visualizer.finalize()
        print(step_visualizer.summary_text())
    output_image = decode_latent_image(pipe, latents, latent_ids, image_size=(width, height))
    if output_image is None:
        raise RuntimeError("failed to decode FLUX.1-Kontext latents into an output image")
    if isinstance(source_image, Image.Image) and output_image.size != source_image.size:
        output_image = output_image.resize(source_image.size, Image.Resampling.LANCZOS)

    maybe_free_model_hooks = getattr(pipe, "maybe_free_model_hooks", None)
    if callable(maybe_free_model_hooks):
        maybe_free_model_hooks()
    return output_image

def run_single_edit(
    *,
    image_path: str | Path,
    prompt: str,
    output_path: str | Path,
    args: argparse.Namespace,
    build_config_fn: Callable[[argparse.Namespace], object] | None = None,
    load_pipeline_fn: Callable[[object], object] | None = None,
    enable_lsedit_fn: Callable[..., object] | None = None,
    load_input_image_fn: Callable[[str | Path], object] | None = None,
    build_pipeline_call_kwargs_fn: Callable[[object, argparse.Namespace, object], dict[str, object]] | None = None,
    extract_output_image_fn: Callable[[object], object] | None = None,
    save_prediction_image_fn: Callable[[object, str | Path], Path] | None = None,
    verify_output_fn: Callable[[object, object, Path], None] | None = None,
) -> Path:
    build_config_fn = build_config_fn or build_generation_config
    load_pipeline_fn = load_pipeline_fn or load_local_pipeline
    enable_lsedit_fn = enable_lsedit_fn or enable_lsedit
    load_input_image_fn = load_input_image_fn or load_input_image
    build_pipeline_call_kwargs_fn = build_pipeline_call_kwargs_fn or build_pipeline_call_kwargs
    extract_output_image_fn = extract_output_image_fn or extract_output_image
    save_prediction_image_fn = save_prediction_image_fn or save_prediction_image

    config = build_config_fn(args)
    pipeline = load_pipeline_fn(config.runtime)
    if args.enable_lsedit:
        pipeline = _enable_lsedit_with_optional_prompt(enable_lsedit_fn, pipeline, config, prompt=prompt)

    source_image = load_input_image_fn(image_path)
    state = getattr(pipeline, "_lsedit_state", None)
    if state is not None:
        state.source_image = source_image.copy()
    step_visualizer = _build_step_visualizer(args, Path(output_path), source_image)
    if step_visualizer is not None:
        if state is not None and hasattr(state, "configure_visualization"):
            state.configure_visualization(visualize_attention=step_visualizer.save_step_attention)

    call_args = argparse.Namespace(**vars(args))
    existing_callback = getattr(call_args, "_step_visualizer_callback", None)
    existing_tensor_inputs = set(getattr(call_args, "_step_visualizer_tensor_inputs", []))
    call_args.prompt = prompt
    callback_candidates: list[Callable[..., dict[str, object]]] = []
    if existing_callback is not None:
        callback_candidates.append(existing_callback)
    if step_visualizer is not None:
        callback_candidates.append(step_visualizer.callback)
        existing_tensor_inputs.add("latents")
    merged_callback = merge_step_callbacks(callback_candidates)
    if merged_callback is not None:
        call_args._step_visualizer_callback = merged_callback
        if existing_tensor_inputs:
            call_args._step_visualizer_tensor_inputs = sorted(existing_tensor_inputs)
    else:
        call_args._step_visualizer_callback = existing_callback
        if existing_tensor_inputs:
            call_args._step_visualizer_tensor_inputs = sorted(existing_tensor_inputs)
    print(config.runtime.backend)       
    if config.runtime.backend == "flux1-kontext":
        # result = _run_single_edit_flux1_kontext_with_image(
        #             pipeline,
        #             **build_pipeline_call_kwargs_fn(pipeline, call_args, source_image))
        result = _run_single_edit_flux1_kontext_with_image(
            pipeline,
            image=source_image,
            num_inference_steps=args.num_inference_steps,
            output_type="pil",
            return_dict=False,
            guidance_scale=args.guidance_scale,
            generator=None,
            prompt=prompt,
            output_path=Path(output_path),
            max_sequence_length = 512,
            max_area=1024**2,
            _auto_resize=True,
            args=args,
        )
    else:
        result = pipeline(**build_pipeline_call_kwargs_fn(pipeline, call_args, source_image))
    output_image = extract_output_image_fn(result)
    output_image, diff_background_mask = maybe_paste_back_background_pixels_from_diff(output_image, source_image, call_args)
    output_image = maybe_paste_back_background_pixels(
        output_image,
        source_image,
        getattr(pipeline, "_lsedit_state", None),
        bool(getattr(args, "enable_background_pixel_pasteback", False)),
    )
    output_image = normalize_output_image_size(output_image, source_image)
    saved_path = save_prediction_image_fn(output_image, output_path)
    if diff_background_mask is not None:
        save_diff_background_mask_visualization(diff_background_mask, saved_path)
    if bool(getattr(args, "save_source_latent_decode", False)):
        save_source_latent_decode_visualization(pipeline, saved_path)
    if bool(getattr(args, "save_source_vae_recon", False)):
        save_source_vae_recon_visualization(pipeline, source_image, saved_path)
    if verify_output_fn is not None:
        verify_output_fn(source_image, output_image, saved_path)
    if step_visualizer is not None:
        step_visualizer.finalize()
        print(step_visualizer.summary_text())
    return saved_path
