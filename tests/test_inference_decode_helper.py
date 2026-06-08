from types import SimpleNamespace
from unittest import mock
import unittest

import torch

from havedit.eval.inference import decode_latent_image


class InferenceDecodeHelperTests(unittest.TestCase):
    def test_decode_latent_image_uses_backend_adapter_when_available(self):
        adapter = mock.Mock()
        adapter.decode_latents.return_value = ["decoded-image"]
        pipeline = SimpleNamespace(_havedit_backend_adapter=adapter)
        latents = torch.randn(1, 2, 16)
        latent_ids = torch.tensor([[[0, 0, 0], [0, 0, 1]]], dtype=torch.int64)

        result = decode_latent_image(pipeline, latents, latent_ids, image_size=(80, 64))

        self.assertEqual(result, "decoded-image")
        adapter.decode_latents.assert_called_once()
        _, passed_latents, run_ctx, output_type, return_dict = adapter.decode_latents.call_args.args
        self.assertTrue(torch.equal(passed_latents, latents))
        self.assertTrue(torch.equal(run_ctx.latent_ids, latent_ids))
        self.assertEqual(run_ctx.image_width, 80)
        self.assertEqual(run_ctx.image_height, 64)
        self.assertEqual(output_type, "pil")
        self.assertTrue(return_dict)


if __name__ == "__main__":
    unittest.main()
