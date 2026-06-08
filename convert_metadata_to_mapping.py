import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


DEFAULT_INPUT = "/data_ljy/ll/dataset/magicbrush/metadata.jsonl"
DEFAULT_OUTPUT = "/data_ljy/ll/dataset/magicbrush/mapping_file.json"


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON") from exc


def convert_record(record: Dict[str, Any], index: int) -> Tuple[str, Dict[str, str]]:
    key = f"{index:012d}"
    value = {
        "image_path": str(record.get("source_img_path", "")),
        "editing_instruction": str(record.get("instruction", "")),
        "editing_type_id": str(record.get("turn_index", "0")),
        "mask": str(record.get("mask_img_path", "")),
        "target_image_path": str(record.get("target_img_path", "")),
    }
    return key, value


def build_mapping(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    mapping = {}
    for index, record in enumerate(records):
        key, value = convert_record(record, index)
        mapping[key] = value
    return mapping


def convert_file(input_path: Path, output_path: Path) -> int:
    mapping = build_mapping(load_jsonl(input_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2)
    return len(mapping)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 MagicBrush metadata.jsonl 转换为 mapping_file.json。"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"输入 metadata.jsonl 路径，默认：{DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出 mapping_file.json 路径，默认：{DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_file(Path(args.input), Path(args.output))
    print(f"转换完成：{count} 条")
    print(f"输出文件：{Path(args.output)}")


if __name__ == "__main__":
    main()