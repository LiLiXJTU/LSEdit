# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/magicbrush \
#     --output-dir /data_ljy/ll/output/results/magicbrush/flux1_havedit_step10_t83_low0.41_gs25_max30 \
#     --gpu-id 3 \
#     --seed 42 \
#     --guidance-scale 2.5 \
#     --threshold 0.83 \
#     --bhc-tau-low 0.53 \
#     --bhc-tau-high 0.83 \
#     --bhc-lambda-max 0.30 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 3 \
#     --hav-steps 10 \
#     --gaussian-sigma 2 \

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/magicbrush \
    --output-dir /data_ljy/ll/output/results/magicbrush/flux1_havedit_step10_t85_low0.55_gs25_max15 \
    --gpu-id 4 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.85 \
    --bhc-tau-low 0.55 \
    --bhc-tau-high 0.85 \
    --bhc-lambda-max 0.15 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 10 \
    --gaussian-sigma 2 \

