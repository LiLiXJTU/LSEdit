import tempfile
import unittest
from pathlib import Path

from PIL import Image

from havedit.eval.inference import normalize_output_image_size, save_prediction_image


class OutputSizeNormalizationTests(unittest.TestCase):
    def test_normalize_output_image_size_resizes_prediction_back_to_source_size(self):
        source = Image.new("RGB", (512, 512), color="white")
        prediction = Image.new("RGB", (1024, 1024), color="black")

        normalized = normalize_output_image_size(prediction, source)

        self.assertEqual(normalized.size, source.size)

    def test_normalize_output_image_size_returns_same_size_when_already_matching(self):
        source = Image.new("RGB", (512, 512), color="white")
        prediction = Image.new("RGB", (512, 512), color="black")

        normalized = normalize_output_image_size(prediction, source)

        self.assertEqual(normalized.size, prediction.size)

    def test_save_prediction_image_persists_resized_output(self):
        source = Image.new("RGB", (512, 512), color="white")
        prediction = Image.new("RGB", (1024, 1024), color="black")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "prediction.jpg"
            normalized = normalize_output_image_size(prediction, source)
            save_prediction_image(normalized, output_path)

            with Image.open(output_path) as saved:
                self.assertEqual(saved.size, (512, 512))


if __name__ == "__main__":
    unittest.main()
