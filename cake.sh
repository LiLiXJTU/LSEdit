# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir output/flux1 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.8 \
#     --bhc-tau-low 0.7 \
#     --bhc-tau-high 0.8 \
#     --bhc-lambda-max 0.15 \
#     --sample-ids 000000000001 \
#     --alpha 1.5 \
#     --beta 1 \
#     --warmup-steps 3 \
#     --hav-steps 10 \
#     --overwrite

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir output/flux1 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.76 \
#     --bhc-tau-low 0.45 \
#     --bhc-tau-high 0.76 \
#     --bhc-lambda-max 0.3 \
#     --sample-ids 000000000001 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 10 \
#     --visualize-steps \
#     --visualize-dir /data_ljy/ll/output/results/PIE_Bench/flux_kontext_dev_havedit_demo/visualization_000000000001 \
#     --overwrite

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir output/flux1 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --sample-ids 000000000025 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 10 \
#     --overwrite

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir output/flux1 \
    --gpu-id 2 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.83 \
    --bhc-tau-low 0.53 \
    --bhc-tau-high 0.83 \
    --bhc-lambda-max 0.41 \
    --sample-ids 000000000025 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 10 \
    --gaussian-sigma 2 \
    --overwrite