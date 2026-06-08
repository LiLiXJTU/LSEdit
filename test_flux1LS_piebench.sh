# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step10_t72_low0.42 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 10 \
#     --gaussian-sigma 2 

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step13_t72_low0.42 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 13 \
#     --gaussian-sigma 2 

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step16_t72_low0.42 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 16 \
#     --gaussian-sigma 2 

# python scripts/run_piebench_batch.py \
#     --backend flux1-kontext \
#     --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
#     --pie-root /data_ljy/ll/dataset/PIE_Bench \
#     --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step15_t72_low0.42 \
#     --gpu-id 4 \
#     --seed 42 \
#     --threshold 0.72 \
#     --bhc-tau-low 0.42 \
#     --bhc-tau-high 0.72 \
#     --bhc-lambda-max 0.3 \
#     --alpha 1 \
#     --beta 1 \
#     --warmup-steps 4 \
#     --hav-steps 15 \
#     --gaussian-sigma 2 
#ours
python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step10_t83_low0.41_gs25 \
    --gpu-id 5 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.83 \
    --bhc-tau-low 0.53 \
    --bhc-tau-high 0.83 \
    --bhc-lambda-max 0.41 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 10 \
    --gaussian-sigma 2 \

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step10_t83_low0.40_gs25 \
    --gpu-id 5 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.83 \
    --bhc-tau-low 0.53 \
    --bhc-tau-high 0.83 \
    --bhc-lambda-max 0.40 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 10 \
    --gaussian-sigma 2 \

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step15_t83_low0.41_gs25 \
    --gpu-id 5 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.83 \
    --bhc-tau-low 0.53 \
    --bhc-tau-high 0.83 \
    --bhc-lambda-max 0.41 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 15 \
    --gaussian-sigma 2 \

python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir /data_ljy/ll/output/results/PIE_Bench/flux1_havedit_step16_t83_low0.41_gs25 \
    --gpu-id 5 \
    --seed 42 \
    --guidance-scale 2.5 \
    --threshold 0.83 \
    --bhc-tau-low 0.53 \
    --bhc-tau-high 0.83 \
    --bhc-lambda-max 0.41 \
    --alpha 1 \
    --beta 1 \
    --warmup-steps 3 \
    --hav-steps 16 \
    --gaussian-sigma 2 \