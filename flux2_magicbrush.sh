# python scripts/run_flux_demo_headwise_subjectbg_subjectrelease.py \
#     --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --output output/flux2/0000003.png \
#     --gpu-id 5 \
#     --seed 42 \
#     --threshold 0.91 \
#     --bhc-tau-low 0.55 \
#     --bhc-tau-high 0.91 \
#     --bhc-lambda-max 0.15 \
#     --image /data_ljy/ll/dataset/magicbrush/annotation_images/images/dev/0000003_source_img.png \
#     --prompt "put an Easter basket on the desk" \

# python scripts/run_flux_demo_headwise_subjectbg_subjectrelease.py \
#     --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
#     --output output/flux2/0000006.png \
#     --gpu-id 5 \
#     --seed 42 \
#     --threshold 0.91 \
#     --bhc-tau-low 0.55 \
#     --bhc-tau-high 0.91 \
#     --bhc-lambda-max 0.15 \
#     --image /data_ljy/ll/dataset/magicbrush/annotation_images/images/dev/0000006_source_img.png \
#     --prompt "Have there be a dolphin jumping out of the water" \

python scripts/run_flux_demo_headwise_subjectbg_subjectrelease.py \
    --model-path /data_ljy/ll/weight/black-forest-labs/FLUX.2-klein-base-9B \
    --output output/flux2/0000001.png \
    --gpu-id 5 \
    --seed 42 \
    --threshold 0.92 \
    --bhc-tau-low 0.61 \
    --bhc-tau-high 0.92 \
    --bhc-lambda-max 0.15 \
    --image /data_ljy/ll/dataset/magicbrush/annotation_images/images/dev/0000001_source_img.png \
    --prompt "put the zebras next to a river" \