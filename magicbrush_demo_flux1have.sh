# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/magicbrush \
#     --output-dir /data_ljy/ll/output/results/magicbrush/flux1_havedit_demo \
#     --gpu-id 2 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 10 \
#     --gaussian-sigma 2 \
#     --sample-ids 000000000204 

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/magicbrush \
#     --output-dir /data_ljy/ll/output/results/magicbrush/flux1_havedit_demo \
#     --gpu-id 2 \
#     --guidance-scale 2.5 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 10 \
#     --gaussian-sigma 2 \
#     --sample-ids 000000000204 \
#     --overwrite

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/magicbrush \
    --output-dir /data_ljy/ll/output/results/magicbrush/flux1_havedit_demo \
    --gpu-id 3 \
    --guidance-scale 2.5 \
    --seed 42 \
    --threshold 0.85 \
    --bhc-tau-low 0.55 \
    --bhc-tau-high 0.85 \
    --bhc-lambda-max 0.15 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 10 \
    --gaussian-sigma 2 \
    --sample-ids 000000000018 \
    --overwrite