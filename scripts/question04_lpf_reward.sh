gpu=$1
w_bit=$2

# Method 1: 직접 더하기
for smoothness in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    python main_quant.py \
        --gpu "$gpu" \
        --mode DMI \
        --wandb \
        --project_name "TBD_MI_LPF_Reward" \
        --dataset /home/mjatwk/data/imagenet/ \
        --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
        --iterations 4000 \
        --w_bit "$w_bit" \
        --a_bit 8 \
        --lpf \
        --lpf_start 0 \
        --lpf_every 50 \
        --cutoff_ratio 0.7 \
        --reward_after_lpf \
        --smoothness "$smoothness"
done

# # Method 2: loss 형태
# for scale in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
#     python main_quant.py \
#         --gpu "$gpu" \
#         --mode DMI \
#         --wandb \
#         --project_name "TBD_MI_LPF_Reward" \
#         --dataset /home/mjatwk/data/imagenet/ \
#         --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
#         --iterations 4000 \
#         --w_bit "$w_bit" \
#         --a_bit 8 \
#         --lpf \
#         --lpf_start 0 \
#         --lpf_every 50 \
#         --cutoff_ratio 0.7 \
#         --reward_after_lpf \
#         --scale_edge "$scale"
# done
