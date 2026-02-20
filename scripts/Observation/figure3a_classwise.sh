gpu=$1

python generate_classwise_imagenet.py \
    --gpu "$gpu" \
    --output_dir "./observation/08_Confidence/DMI_Filtered_0.6/" \
    --dataset /home/mjatwk/data/imagenet/ \
    --synthetic_bs 32 \
    --images_per_class 1 \
    --lpf \
    --lpf_type "gaussian" \
    --lpf_every 50 \
    --cutoff_ratio 0.6
