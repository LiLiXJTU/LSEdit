from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lsedit.eval.inference import load_local_pipeline, run_single_edit
from lsedit.flux.attention_processor_headwise_subjectbg_subjectrelease import (
    enable_headwise_subjectbg_subjectrelease_havedit,
)
from scripts.run_flux_demo_headwise_subjectbg_subjectrelease import (
    build_headwise_subjectbg_subjectrelease_config,
)


DEFAULT_PIE_ROOT = "/workspace/data/PIEbench"
DEFAULT_OUTPUT_DIR = str(REPO_ROOT / "output" / "piebench")


def mode_name_from_backend(backend: str) -> str:
    if backend == "flux1-kontext":
        return "havedit_flux1_kontext"
    if backend == "qwen-image-edit":
        return "havedit_qwen_image_edit"
    return "havedit_flux2_klein_base"


@dataclass(frozen=True)
class PieBenchSample:
    sample_id: str
    image_rel_path: str
    image_path: Path
    prompt: str
    editing_type_id: str
    original_prompt: str | None
    editing_prompt: str | None
    encoded_mask: list[int] | None


def parse_sample_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def strength_to_filename(strength: float) -> str:
    return f"strength_{strength:.2f}".replace(".", "_") + ".jpg"


def build_config_dir_name(backend: str, num_inference_steps: int, guidance_scale: float, seed: int | None) -> str:
    mode = mode_name_from_backend(backend)
    guidance_name = str(guidance_scale).replace(".", "_")
    seed_name = "none" if seed is None else str(seed)
    return f"{mode}_t{num_inference_steps}_g{guidance_name}_seed{seed_name}"


def build_sample_output_dir(output_dir: Path, config_dir_name: str, sample: PieBenchSample) -> Path:
    image_path_dir = Path(sample.image_rel_path).parent
    if str(image_path_dir) == ".":
        return output_dir / config_dir_name / sample.sample_id
    return output_dir / config_dir_name / image_path_dir / sample.sample_id


def build_generated_output_path(sample_output_dir: Path, strength: float = 1.0) -> Path:
    return sample_output_dir / strength_to_filename(strength)


def build_original_output_path(sample_output_dir: Path) -> Path:
    return sample_output_dir / "original.jpg"


def build_metadata_path(sample_output_dir: Path) -> Path:
    return sample_output_dir / "metadata.json"


def build_sample_metadata(
    sample: PieBenchSample,
    args: argparse.Namespace,
    *,
    generated_filename: str,
) -> dict[str, object]:
    mode = mode_name_from_backend(args.backend)
    return {
        "backend": args.backend,
        "image_id": sample.sample_id,
        "original_path": sample.image_rel_path,
        "image_path": sample.image_rel_path,
        "editing_instruction": sample.prompt,
        "editing_type_id": sample.editing_type_id,
        "original_prompt": sample.original_prompt,
        "editing_prompt": sample.editing_prompt,
        "mask": sample.encoded_mask,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "model_path": args.model_path,
        "mode": mode,
        "enable_cpu_offload": bool(args.enable_cpu_offload),
        "enable_bhc": bool(args.enable_bhc),
        "enable_havedit": bool(args.enable_havedit),
        "strengths": {
            "1.00": {
                "mode": mode,
                "blend_weight_a": None,
                "path": generated_filename,
            }
        },
        "havedit": {
            "warmup_steps": args.warmup_steps,
            "alpha": args.alpha,
            "beta": args.beta,
            "threshold": args.threshold,
        },
    }


def save_original_image(source_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as source_image:
        source_image.convert("RGB").save(output_path)
    return output_path


def write_metadata(output_path: Path, metadata: dict[str, object]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def _sample_from_mapping_entry(pie_root: Path, sample_id: str, entry: dict[str, object]) -> PieBenchSample:
    image_rel_path = entry.get("image_path")
    prompt = entry.get("editing_instruction")
    if not isinstance(image_rel_path, str) or not image_rel_path:
        raise ValueError(f"sample {sample_id} is missing a valid image_path")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(f"sample {sample_id} is missing a valid editing_instruction")
    return PieBenchSample(
        sample_id=sample_id,
        image_rel_path=image_rel_path,
        image_path=pie_root / "annotation_images" / image_rel_path,
        prompt=prompt,
        editing_type_id=str(entry.get("editing_type_id", "")),
        original_prompt=str(entry["original_prompt"]) if entry.get("original_prompt") is not None else None,
        editing_prompt=str(entry["editing_prompt"]) if entry.get("editing_prompt") is not None else None,
        encoded_mask=list(entry["mask"]) if isinstance(entry.get("mask"), list) else None,
    )


def select_entries(
    mapping: dict[str, dict[str, object]],
    *,
    sample_ids: list[str] | None,
    start: int,
    limit: int | None,
    pie_root: Path | None = None,
) -> list[PieBenchSample]:
    pie_root = pie_root or Path(DEFAULT_PIE_ROOT)

    if sample_ids:
        ordered_ids = []
        for sample_id in sample_ids:
            if sample_id not in mapping:
                raise KeyError(f"sample id {sample_id!r} was not found in mapping_file.json")
            ordered_ids.append(sample_id)
    else:
        ordered_ids = sorted(mapping.keys())

    if start < 0:
        raise ValueError("start must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    sliced_ids = ordered_ids[start:]
    if limit is not None:
        sliced_ids = sliced_ids[:limit]

    return [_sample_from_mapping_entry(pie_root, sample_id, mapping[sample_id]) for sample_id in sliced_ids]


def load_mapping(pie_root: Path) -> dict[str, dict[str, object]]:
    mapping_path = pie_root / "mapping_file.json"
    return json.loads(mapping_path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HAVEdit on PIEBench samples in batch mode")
    parser.add_argument("--backend", choices=("flux2", "flux1-kontext", "qwen-image-edit"), default="flux2")
    parser.add_argument("--model-path", default="/workspace/model/FLUX.2-klein-base-9B")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.set_defaults(
        enable_cpu_offload=False,
        enable_bhc=True,
        enable_havedit=True,
        save_step_image=False,
        save_step_semantic_prior=True,
        save_step_attention=True,
    )
    parser.add_argument("--enable-cpu-offload", dest="enable_cpu_offload", action="store_true")
    parser.add_argument("--disable-cpu-offload", dest="enable_cpu_offload", action="store_false")
    parser.add_argument("--disable-bhc", dest="enable_bhc", action="store_false")
    parser.add_argument("--disable-havedit", dest="enable_havedit", action="store_false")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--hav-steps", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bhc-tau-low", type=float, default=0.35)
    parser.add_argument("--bhc-tau-high", type=float, default=0.65)
    parser.add_argument("--bhc-lambda-max", type=float, default=0.15)
    parser.add_argument("--visualize-steps", action="store_true")
    parser.add_argument("--visualize-dir", default="/data/ll/output/HAVEdit_PIE_Bench_show")
    parser.add_argument("--visualize-every-n", type=int, default=1)
    parser.add_argument("--disable-save-step-image", dest="save_step_image", action="store_false")
    parser.add_argument("--disable-save-step-semantic-prior", dest="save_step_semantic_prior", action="store_false")
    parser.add_argument("--disable-save-step-attention", dest="save_step_attention", action="store_false")
    parser.add_argument("--local-kernel-size", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=1.0,
                           help="Std of the Gaussian blur applied to the WSP semantic prior "
                                 "(config.wsp.gaussian_sigma). Larger = smoother/blobbier prior.")
    parser.add_argument("--pie-root", default=DEFAULT_PIE_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--sample-ids",
        help="Comma-separated PIEBench sample ids to run, e.g. 000000000005,924000000002",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def run_batch(args: argparse.Namespace) -> int:
    pie_root = Path(args.pie_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir_name = build_config_dir_name(
        backend=args.backend,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    mapping = load_mapping(pie_root)
    sample_ids = parse_sample_ids(args.sample_ids)
    samples = select_entries(
        mapping,
        sample_ids=sample_ids or None,
        start=args.start,
        limit=args.limit,
        pie_root=pie_root,
    )
    if not samples:
        print("No samples selected.")
        return 0

    shared_config = build_headwise_subjectbg_subjectrelease_config(args)
    pipeline = load_local_pipeline(shared_config.runtime)
    if args.enable_havedit:
        pipeline = enable_headwise_subjectbg_subjectrelease_havedit(pipeline, shared_config)

    completed = 0
    skipped = 0
    total = len(samples)
    for index, sample in enumerate(samples, start=1):
        if not sample.image_path.exists():
            raise FileNotFoundError(f"input image not found for sample {sample.sample_id}: {sample.image_path}")

        sample_output_dir = build_sample_output_dir(output_dir, config_dir_name, sample)
        generated_output_path = build_generated_output_path(sample_output_dir)
        original_output_path = build_original_output_path(sample_output_dir)
        metadata_path = build_metadata_path(sample_output_dir)
        metadata = build_sample_metadata(
            sample,
            args,
            generated_filename=generated_output_path.name,
        )

        sample_output_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite or not original_output_path.exists():
            save_original_image(sample.image_path, original_output_path)

        if generated_output_path.exists() and not args.overwrite:
            if not metadata_path.exists():
                write_metadata(metadata_path, metadata)
            print(f"[{index}/{total}] skip {sample.sample_id} -> {generated_output_path} (already exists)")
            skipped += 1
            continue

        print(f"[{index}/{total}] run {sample.sample_id} -> {generated_output_path}")
        run_single_edit(
            image_path=sample.image_path,
            prompt=sample.prompt,
            output_path=generated_output_path,
            args=args,
            build_config_fn=lambda _args: shared_config,
            load_pipeline_fn=lambda _runtime_cfg: pipeline,
            enable_havedit_fn=enable_headwise_subjectbg_subjectrelease_havedit,
        )
        write_metadata(metadata_path, metadata)
        completed += 1

    print(
        f"Finished PIEBench batch: total={total}, completed={completed}, skipped={skipped}, "
        f"output_dir={output_dir / config_dir_name}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # if args.backend == "flux1-kontext":
    #     args.gaussian_sigma = 2
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
