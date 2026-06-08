from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def _iter_generated_images(sample_dir: Path):
    yield from sorted(sample_dir.glob("strength_*.jpg"))
    yield from sorted(sample_dir.glob("strength_*.png"))


def resize_sample_outputs_to_match_original(sample_dir: Path) -> int:
    original_path = sample_dir / "original.jpg"
    if not original_path.exists():
        return 0

    changed = 0
    with Image.open(original_path) as original_image:
        target_size = original_image.size

    for generated_path in _iter_generated_images(sample_dir):
        with Image.open(generated_path) as generated_image:
            if generated_image.size == target_size:
                continue
            resized = generated_image.convert("RGB").resize(target_size, resample=Image.Resampling.LANCZOS)
        resized.save(generated_path)
        changed += 1

    return changed


def resize_tree_outputs_to_match_original(root: Path) -> tuple[int, int]:
    sample_dirs = 0
    changed_images = 0
    for original_path in sorted(root.rglob("original.jpg")):
        sample_dir = original_path.parent
        sample_dirs += 1
        changed_images += resize_sample_outputs_to_match_original(sample_dir)
    return sample_dirs, changed_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Resize generated HAVEdit outputs back to each sample's original image size")
    parser.add_argument("root", help="Root directory containing HAVEdit sample output folders")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.root)
    sample_dirs, changed_images = resize_tree_outputs_to_match_original(root)
    print(f"Scanned {sample_dirs} sample dirs under {root}")
    print(f"Resized {changed_images} generated images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
