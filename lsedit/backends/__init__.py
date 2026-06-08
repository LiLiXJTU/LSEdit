from __future__ import annotations

from importlib import import_module

_BACKENDS = {
    "flux1-kontext": ("lsedit.backends.flux1_kontext", "Flux1KontextBackendAdapter"),
    "flux2": ("lsedit.backends.flux2", "Flux2BackendAdapter"),
    "qwen-image-edit": ("lsedit.backends.qwen_image_edit", "QwenImageEditBackendAdapter"),
}


def get_backend_adapter_class(name: str):
    try:
        module_name, class_name = _BACKENDS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {name}") from exc

    module = import_module(module_name)
    return getattr(module, class_name)


def list_backends():
    return tuple(sorted(_BACKENDS))
