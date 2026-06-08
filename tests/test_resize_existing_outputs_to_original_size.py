import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.resize_existing_outputs_to_original_size import resize_sample_outputs_to_match_original


class ResizeExistingOutputsTests(unittest.TestCase):
    def test_resize_sample_outputs_to_match_original_rewrites_strength_images_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir) / "sample"
            sample_dir.mkdir(parents=True, exist_ok=True)

            original_path = sample_dir / "original.jpg"
            generated_path = sample_dir / "strength_1_00.jpg"
            untouched_path = sample_dir / "metadata.json"

            Image.new("RGB", (512, 512), color="white").save(original_path)
            Image.new("RGB", (1024, 1024), color="black").save(generated_path)
            untouched_path.write_text("{}", encoding="utf-8")

            changed = resize_sample_outputs_to_match_original(sample_dir)

            self.assertEqual(changed, 1)
            with Image.open(generated_path) as generated:
                self.assertEqual(generated.size, (512, 512))
            with Image.open(original_path) as original:
                self.assertEqual(original.size, (512, 512))
            self.assertEqual(untouched_path.read_text(encoding="utf-8"), "{}")


if __name__ == "__main__":
    unittest.main()
