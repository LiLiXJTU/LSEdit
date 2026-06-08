import importlib
import unittest


class ExistingBackendsUnchangedTests(unittest.TestCase):
    def test_backend_registry_names_remain_stable(self):
        backends = importlib.import_module("havedit.backends")
        self.assertEqual(backends.list_backends(), ("flux1-kontext", "flux2", "qwen-image-edit"))

    def test_flux2_backend_name_remains_stable(self):
        backends = importlib.import_module("havedit.backends")
        self.assertEqual(backends.get_backend_adapter_class("flux2").backend_name, "flux2")

    def test_flux1_kontext_backend_name_remains_stable(self):
        backends = importlib.import_module("havedit.backends")
        self.assertEqual(
            backends.get_backend_adapter_class("flux1-kontext").backend_name,
            "flux1-kontext",
        )

    def test_qwen_image_edit_backend_name_is_registered_without_renaming_flux_backends(self):
        backends = importlib.import_module("havedit.backends")
        self.assertEqual(
            backends.get_backend_adapter_class("qwen-image-edit").backend_name,
            "qwen-image-edit",
        )


if __name__ == "__main__":
    unittest.main()
