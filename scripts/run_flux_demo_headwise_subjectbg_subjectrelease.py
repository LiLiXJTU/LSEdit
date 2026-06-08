from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from havedit.eval.inference import load_local_pipeline, run_single_edit
from havedit.flux.attention_processor_headwise_subjectbg_subjectrelease import (
    enable_headwise_subjectbg_subjectrelease_havedit,
)

DEFAULT_MODEL_PATH = "/data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B"
DEFAULT_OUTPUT_PATH = "outputs/havedit_demo.png"
DEFAULT_VISUALIZE_DIR = "/data/ll/output/HAVEdit_PIE_Bench_show"


def build_config(args: argparse.Namespace):
    from havedit import HAVEditConfig

    config = HAVEditConfig()
    config.runtime.backend = args.backend
    config.runtime.model_path = args.model_path
    config.runtime.gpu_id = args.gpu_id
    config.runtime.enable_cpu_offload = args.enable_cpu_offload

    config.bhc.enabled = args.enable_bhc
    config.bhc.tau_low = args.bhc_tau_low
    config.bhc.tau_high = args.bhc_tau_high
    config.bhc.lambda_max = args.bhc_lambda_max

    config.wsp.warmup_steps = args.warmup_steps
    config.wsp.gaussian_sigma = float(getattr(args, "gaussian_sigma", config.wsp.gaussian_sigma))
    config.havsr.alpha = args.alpha
    config.havsr.beta = args.beta
    config.havsr.threshold = args.threshold
    config.hav_steps = args.hav_steps
    return config


def build_parser():
    parser = argparse.ArgumentParser(description="Run one HAVEdit headwise subjectbg subject-release demo edit")
    parser.add_argument("--backend", choices=("flux2", "flux1-kontext", "qwen-image-edit"), default="flux2")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.set_defaults(
        enable_cpu_offload=True,
        enable_bhc=True,
        enable_havedit=True,
        save_step_image=False,
        save_step_semantic_prior=True,
        save_step_attention=True,
    )
    parser.add_argument("--disable-cpu-offload", dest="enable_cpu_offload", action="store_false")
    parser.add_argument("--disable-bhc", dest="enable_bhc", action="store_false")
    parser.add_argument("--disable-havedit", dest="enable_havedit", action="store_false")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--warmup-steps", type=int, default=6)
    parser.add_argument("--hav-steps", type=int, default=28)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--bhc-tau-low", type=float, default=0.35)
    parser.add_argument("--bhc-tau-high", type=float, default=0.65)
    parser.add_argument("--bhc-lambda-max", type=float, default=0.15)
    parser.add_argument("--visualize-steps", action="store_true")
    parser.add_argument("--visualize-dir", default=DEFAULT_VISUALIZE_DIR)
    parser.add_argument("--visualize-every-n", type=int, default=1)
    parser.add_argument("--disable-save-step-image", dest="save_step_image", action="store_false")
    parser.add_argument("--disable-save-step-semantic-prior", dest="save_step_semantic_prior", action="store_false")
    parser.add_argument("--disable-save-step-attention", dest="save_step_attention", action="store_false")
    return parser


def build_headwise_subjectbg_subjectrelease_config(args):
    config = build_config(args)
    if hasattr(args, "local_kernel_size"):
        config.havsr.local_kernel_size = int(args.local_kernel_size)
    return config


def _maybe_prepare_overlay_mask(args: argparse.Namespace, *, pie_root: Path | None = None) -> None:
    if not bool(getattr(args, "visualize_steps", False)):
        return
    if getattr(args, "overlay_mask_path", ""):
        return

    image_path = Path(getattr(args, "image", ""))
    if image_path.name != "source.png":
        return

    sample_id = image_path.parent.name
    if not (sample_id.isdigit() and len(sample_id) == 12):
        return

    pie_root = pie_root or Path("/data/ll/dataset/edit/PIE_Bench")
    mapping_path = pie_root / "mapping_file.json"
    if not mapping_path.exists():
        return

    mapping = json.loads(mapping_path.read_text())
    entry = None
    prompt = str(getattr(args, "prompt", "") or "")
    if prompt:
        for candidate in mapping.values():
            if candidate.get("editing_instruction") == prompt:
                entry = candidate
                break
    if entry is None:
        entry = mapping.get(sample_id)
    if entry is None or "mask" not in entry:
        return

    try:
        with Image.open(image_path) as source_image:
            width, height = source_image.size
    except OSError:
        return

    mask_rle = list(entry["mask"])
    flat = np.zeros(width * height, dtype=np.uint8)
    for start, run_length in zip(mask_rle[0::2], mask_rle[1::2]):
        flat[start:start + run_length] = 255
    mask = flat.reshape(height, width)
    overlay_path = REPO_ROOT / "outputs" / f"{sample_id}_overlay_mask.png"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(overlay_path)
    args.overlay_mask_path = str(overlay_path)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    _maybe_prepare_overlay_mask(args)
    output_path = run_single_edit(
        image_path=args.image,
        prompt=args.prompt,
        output_path=Path(args.output),
        args=args,
        build_config_fn=build_headwise_subjectbg_subjectrelease_config,
        load_pipeline_fn=load_local_pipeline,
        enable_havedit_fn=enable_headwise_subjectbg_subjectrelease_havedit,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
