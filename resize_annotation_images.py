import argparse
from pathlib import Path

from PIL import Image


DEFAULT_IMAGE_DIR = "/data_ljy/ll/dataset/magicbrush/annotation_images/images/dev"


def resize_png_images(image_dir: Path, size: int) -> int:
    if not image_dir.exists():
        raise FileNotFoundError(f"图片目录不存在：{image_dir}")

    count = 0
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for image_path in sorted(image_dir.glob("*.png")):
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((size, size), resampling)
            image.save(image_path)
        count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize PNG images in a folder and overwrite them.")
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"PNG 图片目录，默认：{DEFAULT_IMAGE_DIR}",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="目标尺寸，默认：512，即输出 512x512",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = resize_png_images(Path(args.image_dir), args.size)
    print(f"done: resized {count} png images to {args.size}x{args.size}")


if __name__ == "__main__":
    main()