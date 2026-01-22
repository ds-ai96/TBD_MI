gpu=$1

python generate_classwise_imagenet.py \
    --gpu $1 \
    --dataset /home/mjatwk/data/imagenet/ \
    --output_dir "./observation/DMI/" \
    --images_per_class 1    

# gpu=$1
# nums=(1 2 4 8 16 32 64 128 160)
# for num in ${nums[@]}; do
# python main_quant.py \
#     --gpu "$gpu" \
#     --mode DMI \
#     --wandb \
#     --project_name "DMI_NUM" \
#     --dataset /home/mjatwk/data/imagenet/ \
#     --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
#     --iterations 4000 \
#     --w_bit 8 \
#     --a_bit 8 \
#     --num_runs "$num"
# done
