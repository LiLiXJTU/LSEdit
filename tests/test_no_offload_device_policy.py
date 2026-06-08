import argparse
import unittest
from types import SimpleNamespace

import torch

from havedit.eval import inference
from havedit.flux import model_loader


class _FakePipeline:
    def __init__(self):
        self.offload_gpu_id = None
        self.moved_to = None
        self._execution_device = None
        self.device = None

    def enable_model_cpu_offload(self, gpu_id):
        self.offload_gpu_id = gpu_id
        self._execution_device = torch.device(f"cuda:{gpu_id}")

    def to(self, device):
        self.moved_to = str(device)
        self.device = torch.device(device)
        self._execution_device = self.device
        return self


class NoOffloadDevicePolicyTests(unittest.TestCase):
    def test_configure_pipeline_runtime_uses_cpu_offload_when_enabled(self):
        pipe = _FakePipeline()
        runtime_cfg = SimpleNamespace(enable_cpu_offload=True, gpu_id=0)

        model_loader.configure_pipeline_runtime(
            pipe,
            runtime_cfg,
            execution_device=torch.device("cuda:0"),
        )

        self.assertEqual(pipe.offload_gpu_id, 0)
        self.assertIsNone(pipe.moved_to)

    def test_configure_pipeline_runtime_moves_pipeline_to_gpu_when_offload_disabled(self):
        pipe = _FakePipeline()
        runtime_cfg = SimpleNamespace(enable_cpu_offload=False, gpu_id=0)

        model_loader.configure_pipeline_runtime(
            pipe,
            runtime_cfg,
            execution_device=torch.device("cuda:0"),
        )

        self.assertEqual(pipe.moved_to, "cuda:0")
        self.assertIsNone(pipe.offload_gpu_id)

    def test_resolve_generator_device_prefers_pipeline_execution_device(self):
        args = argparse.Namespace(seed=42, gpu_id=0)
        pipe = _FakePipeline()
        pipe._execution_device = torch.device("cpu")

        device = inference.resolve_generator_device(pipe, args, cuda_available=True)

        self.assertEqual(device, "cpu")

    def test_resolve_generator_device_falls_back_to_gpu_when_pipeline_has_no_device(self):
        args = argparse.Namespace(seed=42, gpu_id=0)
        pipe = SimpleNamespace()

        device = inference.resolve_generator_device(pipe, args, cuda_available=True)

        self.assertEqual(device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
