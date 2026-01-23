gpu=$1

for cutoff_ratio in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    python main_quant.py \
        --gpu "$gpu" \
        --mode TBD_MI \
        --wandb \
        --project_name "Idea1_Filter_Type" \
        --dataset /home/mjatwk/data/imagenet/ \
        --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
        --iterations 4000 \
        --w_bit 4 \
        --a_bit 8 \
        --lpf \
        --lpf_type "gaussian" \
        --lpf_every 100 \
        --cutoff_ratio "$cutoff_ratio"
done