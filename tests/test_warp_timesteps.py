import unittest
from types import SimpleNamespace

import torch

from havedit.flux.warp import _prepare_timesteps


class _Config(dict):
    __getattr__ = dict.get


class WarpTimestepsTests(unittest.TestCase):
    def test_prepare_timesteps_falls_back_to_calculate_shift_for_dynamic_shifting(self):
        recorded = {}

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
            recorded["kwargs"] = kwargs
            recorded["sigmas"] = sigmas
            return torch.tensor([1.0, 0.5]), num_inference_steps

        def fake_calculate_shift(*args):
            recorded["calculate_shift_args"] = args
            return 0.73

        scheduler = SimpleNamespace(
            config=_Config(
                use_dynamic_shifting=True,
                base_image_seq_len=256,
                max_image_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
            ),
            order=1,
            set_timesteps=lambda *args, **kwargs: None,
        )
        pipeline = SimpleNamespace(scheduler=scheduler, _execution_device=torch.device("cpu"))
        base_module = SimpleNamespace(
            retrieve_timesteps=fake_retrieve_timesteps,
            calculate_shift=fake_calculate_shift,
            compute_empirical_mu=None,
        )

        timesteps, num_steps, warmup_steps = _prepare_timesteps(
            pipeline,
            base_module,
            torch.zeros(1, 64, 4),
            2,
            None,
        )

        self.assertEqual(recorded["kwargs"]["mu"], 0.73)
        self.assertEqual(recorded["calculate_shift_args"], (64, 256, 4096, 0.5, 1.15))
        self.assertEqual(recorded["sigmas"].shape[0], 2)
        self.assertEqual(timesteps.shape[0], 2)
        self.assertEqual(num_steps, 2)
        self.assertEqual(warmup_steps, 0)

    def test_prepare_timesteps_prefers_compute_empirical_mu_when_available(self):
        recorded = {}

        def fake_retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, **kwargs):
            recorded["kwargs"] = kwargs
            return torch.tensor([1.0, 0.5]), num_inference_steps

        def fake_compute_empirical_mu(*, image_seq_len, num_steps):
            recorded["compute_args"] = (image_seq_len, num_steps)
            return 0.42

        def fake_calculate_shift(*args):
            recorded["calculate_shift_args"] = args
            return 0.73

        scheduler = SimpleNamespace(
            config=_Config(
                use_dynamic_shifting=True,
                base_image_seq_len=256,
                max_image_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15,
            ),
            order=1,
            set_timesteps=lambda *args, **kwargs: None,
        )
        pipeline = SimpleNamespace(scheduler=scheduler, _execution_device=torch.device("cpu"))
        base_module = SimpleNamespace(
            retrieve_timesteps=fake_retrieve_timesteps,
            calculate_shift=fake_calculate_shift,
            compute_empirical_mu=fake_compute_empirical_mu,
        )

        _prepare_timesteps(
            pipeline,
            base_module,
            torch.zeros(1, 64, 4),
            2,
            None,
        )

        self.assertEqual(recorded["kwargs"]["mu"], 0.42)
        self.assertEqual(recorded["compute_args"], (64, 2))
        self.assertNotIn("calculate_shift_args", recorded)


if __name__ == "__main__":
    unittest.main()
