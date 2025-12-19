gpu=$1
w_bit=$2

python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Optimization_EffiMI" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit "$w_bit" \
    --a_bit 8 \
    --lpf \
    --lpf_start 0 \
    --lpf_every 50 \
    --cutoff_ratio 0.7 \
    --reward_after_lpf \
    --scale_edge 0.7 \
    --sc_center \
    --sc_warmup 0 \
    --sc_every 50 \
    --sc_center_lambda 0.8
