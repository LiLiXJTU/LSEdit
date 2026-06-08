# python scripts/run_piebench_batch.py \
#     --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --pie-root /data_ljy/ll/dataset/magicbrush \
#     --output-dir /data_ljy/ll/output/results/magicbrush/flux2_havedit \
#     --gpu-id 5 \
#     --seed 42 \
#     --threshold 0.91 \
#     --bhc-tau-low 0.61 \
#     --bhc-tau-high 0.91 \
#     --bhc-lambda-max 0.15 \
#     --hav-steps 15 \

python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/magicbrush \
    --output-dir /data_ljy/ll/output/results/magicbrush/flux2_havedit_step10_t91_low0.61 \
    --gpu-id 5 \
    --seed 42 \
    --threshold 0.91 \
    --bhc-tau-low 0.61 \
    --bhc-tau-high 0.91 \
    --bhc-lambda-max 0.15 \
    --hav-steps 10 \

python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/magicbrush \
    --output-dir /data_ljy/ll/output/results/magicbrush/flux2_havedit_step10_t90_low0.60 \
    --gpu-id 5 \
    --seed 42 \
    --threshold 0.90 \
    --bhc-tau-low 0.60 \
    --bhc-tau-high 0.90 \
    --bhc-lambda-max 0.15 \
    --hav-steps 10 \

python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/magicbrush \
    --output-dir /data_ljy/ll/output/results/magicbrush/flux2_havedit_step15_t90_low0.60 \
    --gpu-id 5 \
    --seed 42 \
    --threshold 0.90 \
    --bhc-tau-low 0.60 \
    --bhc-tau-high 0.90 \
    --bhc-lambda-max 0.15 \
    --hav-steps 15 \