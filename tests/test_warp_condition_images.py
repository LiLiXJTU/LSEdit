import types
import unittest
from unittest import mock

from havedit.flux.warp import _prepare_condition_images


class WarpConditionImagesTests(unittest.TestCase):
    def test_flux1_kontext_condition_images_follow_preferred_resolution_path(self):
        image_processor = types.SimpleNamespace(
            get_default_height_width=mock.Mock(return_value=(700, 1300)),
            resize=mock.Mock(side_effect=lambda image, h, w: ("resized", image, h, w)),
            preprocess=mock.Mock(return_value="preprocessed"),
        )
        pipeline_cls = type("FakeKontextPipeline", (), {})
        pipeline_cls.__module__ = "diffusers.pipelines.flux.pipeline_flux_kontext"
        pipeline = pipeline_cls()
        pipeline.transformer = types.SimpleNamespace(_havedit_backend="flux1-kontext")
        pipeline.image_processor = image_processor
        pipeline.vae_scale_factor = 16
        pipeline.latent_channels = 16

        condition_images, height, width = _prepare_condition_images(pipeline, "image", None, None)

        image_processor.resize.assert_called_once_with(["image"], 736, 1376)
        image_processor.preprocess.assert_called_once_with(("resized", ["image"], 736, 1376), 736, 1376)
        self.assertEqual(condition_images, "preprocessed")
        self.assertEqual(height, 736)
        self.assertEqual(width, 1376)


if __name__ == "__main__":
    unittest.main()
