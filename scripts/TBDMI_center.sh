gpu=$1
w_bit=$2

# for lambda in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for lambda in 0.0 1.2 1.4 1.6 1.8 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "TBDMI_Idea1_LPF" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit "$w_bit" \
    --a_bit 8 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda "$lambda"
done
