import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from havedit.backends.flux2 import Flux2BackendAdapter


class Flux2AdapterTests(unittest.TestCase):
    def test_check_inputs_preserves_flux2_signature(self):
        adapter = Flux2BackendAdapter()
        pipeline = mock.Mock()
        params = SimpleNamespace(
            prompt="turn the cat black",
            height=64,
            width=80,
            prompt_embeds=None,
            guidance_scale=4.0,
        )

        adapter.check_inputs(pipeline, params, ["latents"])

        pipeline.check_inputs.assert_called_once_with(
            prompt="turn the cat black",
            height=64,
            width=80,
            prompt_embeds=None,
            callback_on_step_end_tensor_inputs=["latents"],
            guidance_scale=4.0,
        )

    def test_decode_latents_prefers_unpack_with_ids_when_available(self):
        adapter = Flux2BackendAdapter()
        bn = SimpleNamespace(
            running_mean=torch.zeros(4),
            running_var=torch.ones(4),
        )
        decoded = torch.randn(1, 3, 8, 8)
        pipeline = SimpleNamespace(
            vae_scale_factor=16,
            _unpack_latents_with_ids=mock.Mock(return_value=torch.randn(1, 4, 2, 2)),
            _unpatchify_latents=mock.Mock(return_value=torch.randn(1, 4, 8, 8)),
            vae=SimpleNamespace(
                bn=bn,
                config=SimpleNamespace(batch_norm_eps=1e-5, scaling_factor=0.5, shift_factor=0.1),
                decode=mock.Mock(return_value=(decoded,)),
            ),
            image_processor=SimpleNamespace(postprocess=mock.Mock(return_value=["image"])),
        )
        run_ctx = SimpleNamespace(latent_ids=torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64))
        latents = torch.randn(1, 2, 16)

        result = adapter.decode_latents(pipeline, latents, run_ctx, "pil", True)

        pipeline._unpack_latents_with_ids.assert_called_once_with(latents, run_ctx.latent_ids)
        pipeline._unpatchify_latents.assert_called_once()
        pipeline.image_processor.postprocess.assert_called_once_with(decoded, output_type="pil")
        self.assertEqual(result, ["image"])


if __name__ == "__main__":
    unittest.main()
