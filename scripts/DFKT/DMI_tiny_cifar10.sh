gpu=$1

# Idea 1: Low-pass filter
python main_kt.py \
    --gpu "$gpu" \
    --mode DMI \
    --model "deit_tiny_16_cifar10" \
    --project_name "DFKT" \
    --dataset /home/jener05458/data/ \
    --model_path /home/jener05458/src/EdgeMI/TBD_MI/models/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/kt_dataset/ \
    --kr-batchsize 128 \
    --prune_it -1 \
    --prune_ratio 0 \
    --wandb \
    
