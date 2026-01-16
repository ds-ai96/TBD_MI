gpu=$1

python generate_classwise_imagenet.py \
    --output_dir /home/jener05458/src/EdgeMI/TBD_MI/observation/05_Gradient/ \
    --gpu "$gpu" \
    --dataset /home/mjatwk/data/imagenet/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8
