gpu=$1

python generate_classwise_imagenet.py \
    --gpu $1 \
    --dataset /home/mjatwk/data/imagenet/ \
    --output_dir "./observation/08_Confidence/DMI/" \
    --images_per_class 1    
