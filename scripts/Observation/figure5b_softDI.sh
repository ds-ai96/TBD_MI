gpu=$1

python generate_classwise_imagenet.py \
    --gpu $1 \
    --dataset /home/mjatwk/data/imagenet/ \
    --output_dir "./observation/08_Confidence/SMI/" \
    --images_per_class 1 \
    --use_soft_label \
    --soft_label_alpha 0.6 \
