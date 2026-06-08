#!/usr/bin/env bash
# Run 2-backbone × 3-ablation PIE-Bench matrix:
#   {FLUX.2-klein, FLUX.1-Kontext-dev} × {baseline, HAVEdit w/o BHC, HAVEdit full}
#
# Usage:
#   bash scripts/run_dual_backbone_ablation.sh                      # full PIE-Bench
#   bash scripts/run_dual_backbone_ablation.sh --limit 3            # smoke: 3 samples per cell
#   bash scripts/run_dual_backbone_ablation.sh --sample-ids 000000000005,924000000002
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

FLUX2_PATH="${FLUX2_PATH:-/data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B}"
FLUX1K_PATH="${FLUX1K_PATH:-/data/ll/weight/black-forest-labs/FLUX.1-Kontext-dev}"
PIE_ROOT="${PIE_ROOT:-/data/ll/data/PIEbench}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/ablation_dual_backbone}"

EXTRA=("$@")

run_one() {
    local label="$1" backend="$2" model_path="$3"
    shift 3
    echo "==> [${label}] backend=${backend} model=${model_path}"
    python scripts/run_piebench_batch.py \
        --backend "${backend}" \
        --model-path "${model_path}" \
        --pie-root "${PIE_ROOT}" \
        --output-dir "${OUTPUT_ROOT}/${label}" \
        --disable-trajectory-trust \
        "${EXTRA[@]}" \
        "$@"
}

run_one "flux2_baseline"        flux2          "${FLUX2_PATH}"  --disable-havedit
run_one "flux2_havedit_no_bhc"  flux2          "${FLUX2_PATH}"  --disable-bhc
run_one "flux2_havedit_full"    flux2          "${FLUX2_PATH}"
run_one "flux1k_baseline"       flux1-kontext  "${FLUX1K_PATH}" --disable-havedit
run_one "flux1k_havedit_no_bhc" flux1-kontext  "${FLUX1K_PATH}" --disable-bhc
run_one "flux1k_havedit_full"   flux1-kontext  "${FLUX1K_PATH}"

echo "==> All 6 cells finished. Output: ${OUTPUT_ROOT}"
