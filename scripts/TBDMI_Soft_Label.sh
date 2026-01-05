gpu=$1
w_bit=$2

for alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "TBDMI_Idea3_Soft_Label" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit "$w_bit" \
    --a_bit 8 \
    --use_soft_label \
    --soft_label_alpha "$alpha"
done
