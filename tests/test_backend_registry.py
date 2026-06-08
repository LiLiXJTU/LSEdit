from contextlib import contextmanager
import importlib
import sys
import unittest

from havedit import HAVEditConfig

REQUIRED_BACKEND_METHODS = (
    "load_pipeline",
    "iter_transformer_blocks",
    "build_run_context",
    "build_step_context",
    "run_cond_step",
    "run_uncond_step",
    "scheduler_step",
    "decode_latents",
)


@contextmanager
def temporarily_remove_cached_modules(*module_names):
    """Temporarily remove module entries from sys.modules and restore them afterward."""
    original_modules = {module_name: sys.modules.get(module_name) for module_name in module_names}

    try:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
        yield
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)
            original_module = original_modules[module_name]
            if original_module is not None:
                sys.modules[module_name] = original_module


class BackendRegistryTests(unittest.TestCase):
    def test_runtime_config_defaults_to_flux2(self):
        config = HAVEditConfig()
        self.assertEqual(config.runtime.backend, "flux2")

    def test_list_backends_contains_flux2_flux1_kontext_and_qwen_image_edit(self):
        backends = importlib.import_module("havedit.backends")
        self.assertEqual(backends.list_backends(), ("flux1-kontext", "flux2", "qwen-image-edit"))

    def test_lookup_returns_distinct_adapter_classes(self):
        backends = importlib.import_module("havedit.backends")
        flux2_cls = backends.get_backend_adapter_class("flux2")
        kontext_cls = backends.get_backend_adapter_class("flux1-kontext")

        self.assertNotEqual(flux2_cls, kontext_cls)

    def test_lookup_rejects_unknown_backend(self):
        backends = importlib.import_module("havedit.backends")
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            backends.get_backend_adapter_class("made-up")

    def test_lookup_returns_qwen_image_edit_adapter(self):
        backends = importlib.import_module("havedit.backends")
        qwen_cls = backends.get_backend_adapter_class("qwen-image-edit")

        self.assertEqual(qwen_cls.backend_name, "qwen-image-edit")

    def test_lazy_registry_does_not_import_adapters_until_lookup(self):
        module_names = ("havedit.backends.flux2", "havedit.backends.flux1_kontext")
        with temporarily_remove_cached_modules(*module_names):
            backends = importlib.import_module("havedit.backends")

            self.assertNotIn("havedit.backends.flux2", sys.modules)
            self.assertNotIn("havedit.backends.flux1_kontext", sys.modules)
            self.assertEqual(backends.list_backends(), ("flux1-kontext", "flux2", "qwen-image-edit"))
            self.assertNotIn("havedit.backends.flux2", sys.modules)
            self.assertNotIn("havedit.backends.flux1_kontext", sys.modules)

            adapter_cls = backends.get_backend_adapter_class("flux2")

            self.assertEqual(adapter_cls.backend_name, "flux2")
            self.assertIn("havedit.backends.flux2", sys.modules)

    def test_flux2_adapter_exposes_required_backend_methods(self):
        backends = importlib.import_module("havedit.backends")
        flux2_cls = backends.get_backend_adapter_class("flux2")
        adapter = flux2_cls()

        for method_name in REQUIRED_BACKEND_METHODS:
            with self.subTest(method_name=method_name):
                self.assertTrue(callable(getattr(adapter, method_name)))

    def test_backend_lookup_instantiates_distinct_adapters(self):
        backends = importlib.import_module("havedit.backends")
        adapter_classes = []

        for backend_name in ("flux2", "flux1-kontext", "qwen-image-edit"):
            adapter = backends.get_backend_adapter_class(backend_name)()
            adapter_classes.append(type(adapter))

        self.assertEqual(len(set(adapter_classes)), 3)


if __name__ == "__main__":
    unittest.main()
