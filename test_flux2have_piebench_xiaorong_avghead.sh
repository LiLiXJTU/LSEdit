#在代码里添加了preserve_score = preserve_score.mean(dim=1, keepdim=True).expand_as(preserve_score)
# if self.state.config.havsr.decision_granularity == "token":
        # z_dev = z_dev.mean(dim=1, keepdim=True).expand_as(z_dev)
python scripts/run_piebench_batch.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --pie-root /data_ljy/ll/dataset/PIE_Bench \
    --output-dir /data_ljy/ll/output/results/PIE_Bench/flux2_havedit_step10_t90_low0.60_avghead_z \
    --gpu-id 4 \
    --seed 42 \
    --threshold 0.90 \
    --bhc-tau-low 0.60 \
    --bhc-tau-high 0.90 \
    --bhc-lambda-max 0.15 \
    --hav-steps 10 \
    --warmup-steps 6