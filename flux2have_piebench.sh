#ours
python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir output/flux2/show_git \
    --gpu-id 4 \
    --seed 42 \
    --threshold 0.90 \
    --bhc-tau-low 0.60 \
    --bhc-tau-high 0.90 \
    --bhc-lambda-max 0.15 \
    --hav-steps 10 \
    --sample-ids 000000000025 \
    --overwrite