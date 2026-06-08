import argparse
import importlib.util
import sys
import unittest
from pathlib import Path

from havedit.eval.inference import build_generation_config


DEMO_SCRIPT_PATH = Path(
    "/workspace/code/HAVEdit-main_changed/scripts/run_flux_demo_headwise_subjectbg_subjectrelease.py"
)


def load_demo_module():
    spec = importlib.util.spec_from_file_location("run_flux_demo_headwise_subjectbg_subjectrelease", DEMO_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BackendCliConfigTests(unittest.TestCase):
    def setUp(self):
        self.demo_module = load_demo_module()

    def test_demo_parser_defaults_backend_to_flux2(self):
        parser = self.demo_module.build_parser()

        args = parser.parse_args(["--image", "/tmp/source.png", "--prompt", "hello"])

        self.assertEqual(args.backend, "flux2")

    def test_demo_build_config_threads_backend_into_runtime_config(self):
        args = argparse.Namespace(
            backend="flux1-kontext",
            model_path="/workspace/model/FLUX.1-Kontext-dev",
            gpu_id=0,
            enable_cpu_offload=False,
            enable_bhc=True,
            bhc_tau_low=0.35,
            bhc_tau_high=0.65,
            bhc_lambda_max=0.15,
            warmup_steps=6,
            alpha=2.0,
            beta=1.0,
            threshold=0.9,
            enable_trajectory_trust=True,
        )

        config = self.demo_module.build_config(args)

        self.assertEqual(config.runtime.backend, "flux1-kontext")
        self.assertEqual(config.runtime.model_path, "/workspace/model/FLUX.1-Kontext-dev")

    def test_demo_parser_accepts_qwen_backend(self):
        parser = self.demo_module.build_parser()

        args = parser.parse_args(
            [
                "--image",
                "/tmp/source.png",
                "--prompt",
                "hello",
                "--backend",
                "qwen-image-edit",
                "--model-path",
                "/workspace/model/Qwen-Image-Edit",
            ]
        )

        self.assertEqual(args.backend, "qwen-image-edit")
        self.assertEqual(args.model_path, "/workspace/model/Qwen-Image-Edit")

    def test_build_generation_config_threads_backend_into_runtime_config(self):
        args = argparse.Namespace(
            backend="flux1-kontext",
            model_path="/workspace/model/FLUX.1-Kontext-dev",
            gpu_id=1,
            enable_cpu_offload=True,
            enable_bhc=True,
            warmup_steps=6,
            alpha=2.0,
            beta=1.0,
            threshold=0.9,
        )

        config = build_generation_config(args)

        self.assertEqual(config.runtime.backend, "flux1-kontext")
        self.assertEqual(config.runtime.model_path, "/workspace/model/FLUX.1-Kontext-dev")
        self.assertEqual(config.runtime.gpu_id, 1)

    def test_pipeline_call_kwargs_maps_guidance_to_qwen_true_cfg(self):
        from havedit.eval.inference import build_pipeline_call_kwargs

        class FakeQwenPipeline:
            def __call__(
                self,
                image=None,
                prompt=None,
                true_cfg_scale=4.0,
                negative_prompt=None,
                num_inference_steps=1,
                output_type="pil",
                return_dict=True,
                generator=None,
            ):
                return None

        args = argparse.Namespace(
            prompt="make the car red",
            num_inference_steps=3,
            guidance_scale=5.5,
            seed=None,
            gpu_id=0,
        )

        kwargs = build_pipeline_call_kwargs(FakeQwenPipeline(), args, object())

        self.assertEqual(kwargs["true_cfg_scale"], 5.5)
        self.assertEqual(kwargs["negative_prompt"], "")
        self.assertNotIn("guidance_scale", kwargs)


if __name__ == "__main__":
    unittest.main()
