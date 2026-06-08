python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/kontext-bench \
    --output-dir /data_ljy/ll/output/results/kontext-bench/flux2_nohavedit \
    --gpu-id 5 \
    --seed 42 \
    --disable-havedit \
    --overwrite