#!/usr/bin/env bash
# set -euo pipefail
    # --trajectory-trust-ema-decay 0.7 \
    # --trajectory-trust-release-bias 1 \
    # --trajectory-trust-release-scale 1 \
    # --trajectory-trust-min-steps 6 \
# REPO_ROOT="/home/ll/vibe/code/HAVEdit-main"
# PYTHON_BIN="/data/ll/envs/diff3.10/bin/python"
# VIS_DIR="outputs/cartoon_no_tt_auto_steps"
# # Current CLI default threshold is already 0.9.
# # This launcher keeps the historical output name without inventing a second scenario.
# cd "$REPO_ROOT"

# "$PYTHON_BIN" scripts/run_flux_demo_trajectory_trust.py \
#     --model-path /data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --gpu-id 0 \
#     --image /data/ll/output/HAVEdit_PIE_Bench/924000000002/source.png \
#     --prompt "Add cartoon effect" \
#     --seed 42 \
#     --threshold 0.99 \
#     --warmup-steps 6 \
#     --disable-trajectory-trust \
#     --print-step-deviation-debug \
#     --print-step-preserve-weight-debug \
#     --print-step-consistency-score-debug \
#     --print-step-release-score-debug \
#     --print-step-trust-score-debug \
#     --track-step-deviations \
#     --step-deviation-json outputs/cartoon_deviation_99.json \
#     --output outputs/cartoon_99.png

# "$PYTHON_BIN" scripts/run_flux_demo_trajectory_trust.py \
#     --model-path /data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --gpu-id 0 \
#     --image /data/ll/output/HAVEdit_PIE_Bench/924000000002/source.png \
#     --prompt "Add cartoon effect" \
#     --seed 42 \
#     --threshold 0.9 \
#     --disable-trajectory-trust \
#     --print-step-deviation-debug \
#     --print-step-preserve-weight-debug \
#     --print-step-consistency-score-debug \
#     --print-step-release-score-debug \
#     --print-step-trust-score-debug \
#     --track-step-deviations \
#     --step-deviation-json outputs/cartoon_deviation_no_tt.json \
#     --output outputs/cartoon_no_tt.png

# "$PYTHON_BIN" scripts/run_flux_demo_softmap.py \
#     --model-path /data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --gpu-id 0 \
    # --image /data/ll/output/HAVEdit_PIE_Bench/000000000025/source.png \
    # --prompt "change the woman to a storm-trooper" \
#     --seed 42 \
#     --disable-trajectory-trust \
#     --softmap-source semantic \
#     --softmap-tau-core 0.9 \
#     --softmap-ring-radius 9 \
#     --softmap-confusion-gamma 2.0 \
#     --visualize-steps \
#     --visualize-dir "$VIS_DIR" \
#     --output outputs/cartoon_no_tt_semantic.png
    # --visualize-steps \
    # --visualize-dir /home/ll/vibe/code/HAVEdit/outputs/black_no_tt_subjectbg_frozen16_steps \
# "$PYTHON_BIN" scripts/run_flux_demo_headwise_subjectbg_latentreplace.py \
#     --model-path /data/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --gpu-id 0 \
#     --image /data/ll/output/HAVEdit_PIE_Bench/000000000005/source.png \
#     --prompt "Change the color of the cat from orange to black" \
#     --seed 42 \
#     --disable-trajectory-trust \
#     --threshold 0.91 \
#     --subject-threshold 0.17 \
#     --subject-select-mode largest \
#     --subject-open-kernel 3 \
#     --subject-close-kernel 1 \
#     --subject-dilate-radius 1 \
#     --background-discovery-step 16 \
#     --disable-background-latent-replace \
#     --enable-background-pixel-pasteback \
#     --background-pixel-ring-width 2 \
#     --background-pixel-ring-alpha 0.6 \
#     --output outputs/black_no_tt_subjectbg_latentreplace.png
    # --visualize-steps \
    # --visualize-dir /home/ll/vibe/code/HAVEdit/outputs/black_no_tt_subjectbg_latentreplace_steps \
python scripts/run_flux_demo_headwise_subjectbg_subjectrelease.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --gpu-id 5 \
    --image /data_ljy/ll/dataset/PIE_Bench/annotation_images/0_random_140/000000000005.jpg \
    --prompt "Change the color of the cat from orange to black" \
    --seed 42 \
    --disable-trajectory-trust \
    --threshold 0.91 \
    --subject-threshold 0.17 \
    --subject-select-mode largest \
    --subject-open-kernel 3 \
    --subject-close-kernel 1 \
    --subject-dilate-radius 1 \
    --background-discovery-step 16 \
    --subject-release-step 16 \
    --subject-release-scale 0 \
    --output outputs/5.png
