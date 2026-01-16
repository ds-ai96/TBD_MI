gpu=$1
w_bit=$2

# for alpha in 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00; do
# python main_quant.py \
#     --gpu "$gpu" \
#     --mode DMI \
#     --wandb \
#     --project_name "DMI_Soft_label" \
#     --dataset /home/mjatwk/data/imagenet/ \
#     --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
#     --iterations 4000 \
#     --w_bit "$w_bit" \
#     --a_bit 8 \
#     --seed 0 \
#     --use_soft_label \
#     --soft_label_alpha "$alpha"
# done

for alpha in 0.60 0.65 0.70 0.75 0.80; do
python main_quant.py \
    --gpu "$gpu" \
    --mode SMI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit "$w_bit" \
    --a_bit 8 \
    --seed 0 \
    --use_soft_label \
    --soft_label_alpha "$alpha"
done