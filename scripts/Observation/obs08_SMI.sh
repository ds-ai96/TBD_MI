gpu=$1

python generate_classwise_imagenet.py \
    --gpu $1 \
    --dataset /home/mjatwk/data/imagenet/ \
    --output_dir "./observation/08_Confidence/SMI/" \
    --images_per_class 1 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
