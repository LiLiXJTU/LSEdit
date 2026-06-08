python scripts/run_piebench_batch.py \
    --backend flux1-kontext \
    --model-path /data_ljy/ll/weight/FLUX.1-Kontext-dev \
    --pie-root /data_ljy/ll/dataset/kontext-bench \
    --output-dir /data_ljy/ll/output/results/kontext-bench/flux1_kontext_dev_nohave_gs25\
    --disable-havedit \
    --guidance-scale 2.5 \
    --gpu-id 4 \
    --seed 42 \
    --overwrite