import argparse
import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


COMMON_IMAGE_COLUMN_HINTS = (
    "image",
    "img",
    "source",
    "target",
    "mask",
    "edited",
    "input",
    "output",
)


def is_image_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bytes, bytearray)):
        return True
    if isinstance(value, dict):
        raw = value.get("bytes")
        path = value.get("path")
        return isinstance(raw, (bytes, bytearray)) or (
            isinstance(path, str)
            and path.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        )
    return hasattr(value, "save") and hasattr(value, "mode")


def detect_image_columns(row: Dict[str, Any]) -> List[str]:
    image_columns = []
    for column, value in row.items():
        lower = column.lower()
        hinted = any(hint in lower for hint in COMMON_IMAGE_COLUMN_HINTS)
        if hinted and is_image_value(value):
            image_columns.append(column)
    return image_columns


def image_bytes_from_value(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    raise ValueError("unsupported image value")


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)


def export_row_images(
    row: Dict[str, Any],
    image_columns: Iterable[str],
    output_dir: Path,
    split_name: str,
    row_index: int,
) -> Dict[str, str]:
    if Image is None:
        raise RuntimeError("缺少 Pillow，请先安装：pip install pillow")

    exported = {}
    image_dir = output_dir / "images" / safe_name(split_name)
    image_dir.mkdir(parents=True, exist_ok=True)

    for column in image_columns:
        value = row.get(column)
        if not is_image_value(value):
            continue
        raw = image_bytes_from_value(value)
        image = Image.open(io.BytesIO(raw))
        filename = f"{row_index:07d}_{safe_name(column)}.png"
        path = image_dir / filename
        image.save(path)
        exported[column] = path.relative_to(output_dir).as_posix()

    return exported


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [normalize_scalar(item) for item in value]
    if isinstance(value, dict):
        if set(value.keys()).issubset({"bytes", "path"}):
            return None
        return {key: normalize_scalar(item) for key, item in value.items()}
    return str(value)


def build_metadata_record(
    row: Dict[str, Any],
    image_columns: Iterable[str],
    exported_paths: Dict[str, str],
    source_file: str,
    row_index: int,
) -> Dict[str, Any]:
    image_column_set = set(image_columns)
    record = {
        "source_file": source_file,
        "row_index": row_index,
    }

    for column, value in row.items():
        if column in image_column_set:
            continue
        normalized = normalize_scalar(value)
        if normalized is not None:
            record[column] = normalized

    for column, path in exported_paths.items():
        record[f"{column}_path"] = path

    return record


def split_name_from_file(path: Path) -> str:
    return path.name.split("-", 1)[0] if "-" in path.name else path.stem


def iter_parquet_rows(parquet_path: Path) -> Iterable[Dict[str, Any]]:
    if pd is None:
        raise RuntimeError("缺少 pandas，请先安装：pip install pandas pyarrow")
    try:
        frame = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise RuntimeError("读取 parquet 需要 pyarrow 或 fastparquet：pip install pyarrow") from exc
    for row in frame.to_dict(orient="records"):
        yield row


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    columns = sorted({column for record in records for column in record.keys()})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    image_columns: Optional[List[str]] = None,
    write_csv_file: bool = True,
) -> List[Dict[str, Any]]:
    parquet_files = sorted(input_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"没有找到 parquet 文件：{input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    global_index = 0

    for parquet_path in parquet_files:
        split_name = split_name_from_file(parquet_path)
        rows = iter_parquet_rows(parquet_path)
        detected_columns = image_columns

        for row in rows:
            if detected_columns is None:
                detected_columns = detect_image_columns(row)
            exported = export_row_images(
                row=row,
                image_columns=detected_columns,
                output_dir=output_dir,
                split_name=split_name,
                row_index=global_index,
            )
            records.append(
                build_metadata_record(
                    row=row,
                    image_columns=detected_columns,
                    exported_paths=exported,
                    source_file=parquet_path.name,
                    row_index=global_index,
                )
            )
            global_index += 1

    write_jsonl(records, output_dir / "metadata.jsonl")
    if write_csv_file:
        write_csv(records, output_dir / "metadata.csv")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量处理 MagicBrush parquet 数据集，导出图片并生成元数据。"
    )
    parser.add_argument(
        "--input-dir",
        default="/data_ljy/ll/dataset/magicbrush",
        help="parquet 数据集目录，默认：/data_ljy/ll/dataset/magicbrush",
    )
    parser.add_argument(
        "--output-dir",
        default="/data_ljy/ll/dataset/magicbrush",
        help="输出目录，默认：./magicbrush_processed",
    )
    parser.add_argument(
        "--image-columns",
        nargs="*",
        default=None,
        help="手动指定图片列名；不填则自动识别常见图片列。",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="只输出 metadata.jsonl，不输出 metadata.csv。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = process_dataset(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        image_columns=args.image_columns,
        write_csv_file=not args.no_csv,
    )
    print(f"处理完成：{len(records)} 条样本")
    print(f"输出目录：{Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
