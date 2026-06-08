#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/code/HAVEdit-main_changed/output/piebench_black_subjectrelease_nooffload_seed42_20260417"
CONFIG_DIR="$ROOT/havedit_flux2_klein_base_t28_g4_0_seed42"
RESULTS_LINK_ROOT="$ROOT/.veloedit_eval_input"
OUTPUT_DIR="$ROOT/eval_all_metrics"
LOG_FILE="$ROOT/eval_all_metrics.log"

VELOEDIT_ROOT="/workspace/code/VeloEdit"
BENCHMARK_ROOT="/workspace/data/PIEbench"
PYTHON_BIN="/opt/conda/envs/flux2/bin/python"

export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-160}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-160}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-160}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-160}"
export TORCH_HOME="${TORCH_HOME:-/root/.cache/torch}"
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$RESULTS_LINK_ROOT" "$OUTPUT_DIR"
ln -sfn "$CONFIG_DIR" "$RESULTS_LINK_ROOT/$(basename "$CONFIG_DIR")"

cd "$VELOEDIT_ROOT"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] START VeloEdit all-metrics evaluation on HAVEdit outputs"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] benchmark_path=$BENCHMARK_ROOT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] results_path=$RESULTS_LINK_ROOT"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] output_dir=$OUTPUT_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] log_file=$LOG_FILE"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] cpu_threads=$OMP_NUM_THREADS"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] PREWARM CLIP and DINO caches"
"$PYTHON_BIN" -u - <<'PY'
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/code/VeloEdit")

from huggingface_hub import snapshot_download

from evaluation.metrics import MetricCalculator

clip_repo = "openai/clip-vit-large-patch14"
clip_cache_root = Path("/root/.cache/huggingface/hub/models--openai--clip-vit-large-patch14")
required_clip_files = {
    ".gitattributes",
    "README.md",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
if clip_cache_root.exists():
    snapshots = list((clip_cache_root / "snapshots").glob("*"))
    has_required_snapshot = any(all((snapshot / name).exists() for name in required_clip_files if name != ".gitattributes") for snapshot in snapshots)
    has_extra_framework_weights = any(
        (snapshot / "pytorch_model.bin").exists() or (snapshot / "tf_model.h5").exists() or (snapshot / "flax_model.msgpack").exists()
        for snapshot in snapshots
    )
    if (not has_required_snapshot) or has_extra_framework_weights:
        shutil.rmtree(clip_cache_root)

local_path = Path(
    snapshot_download(
        repo_id=clip_repo,
        allow_patterns=sorted(required_clip_files),
    )
)
preprocessor_config = local_path / "preprocessor_config.json"
processor_config = local_path / "processor_config.json"
if preprocessor_config.exists() and not processor_config.exists():
    shutil.copyfile(preprocessor_config, processor_config)
print(f"CLIP snapshot ready at {local_path}")

calculator = MetricCalculator(device="cpu")
calculator._get_clip_metric()
print("CLIP cache ready")
calculator._get_structure_distance_metric()
print("DINO cache ready")
PY

"$PYTHON_BIN" -u evaluate.py \
  --benchmark-path "$BENCHMARK_ROOT" \
  --results-path "$RESULTS_LINK_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --metrics \
    psnr lpips mse ssim structure_distance \
    psnr_unedit_part lpips_unedit_part mse_unedit_part ssim_unedit_part structure_distance_unedit_part \
    psnr_edit_part lpips_edit_part mse_edit_part ssim_edit_part structure_distance_edit_part \
    clip_similarity_source_image clip_similarity_target_image clip_similarity_target_image_edit_part \
  --device cpu

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] DONE"
