from __future__ import annotations

import torch

from lsedit.backends import get_backend_adapter_class


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def resolve_runtime_device(runtime_cfg, *, cuda_available: bool | None = None) -> torch.device:
    if cuda_available is None:
        cuda_available = torch.cuda.is_available()
    gpu_id = getattr(runtime_cfg, "gpu_id", None)
    if gpu_id is not None and cuda_available:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def maybe_enable_cpu_offload(pipeline, runtime_cfg):
    if runtime_cfg.enable_cpu_offload:
        pipeline.enable_model_cpu_offload(gpu_id=runtime_cfg.gpu_id)
    return pipeline


def configure_pipeline_runtime(pipeline, runtime_cfg, *, execution_device: torch.device | None = None):
    execution_device = execution_device or resolve_runtime_device(runtime_cfg)
    if runtime_cfg.enable_cpu_offload and execution_device.type == "cuda":
        pipeline.enable_model_cpu_offload(gpu_id=runtime_cfg.gpu_id)
        return pipeline
    return pipeline.to(str(execution_device))


def load_local_pipeline(runtime_cfg):
    adapter_cls = get_backend_adapter_class(runtime_cfg.backend)
    adapter = adapter_cls()
    return adapter.load_pipeline(runtime_cfg)
