gpu=$1

for cutoff_ratio in 0.5 0.6 0.7 0.8 0.9; do
# DeepInversion with Low-Pass Filter
# python main_quant.py \
#     --gpu "$gpu" \
#     --mode TBD_MI \
#     --wandb \
#     --project_name "TBDMI_Idea1_LPF" \
#     --dataset /home/mjatwk/data/imagenet/ \
#     --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
#     --iterations 4000 \
#     --w_bit 4 \
#     --a_bit 8 \
#     --lpf \
#     --lpf_every 100 \
#     --cutoff_ratio "$cutoff_ratio"

# # Sparse Model Inversion with Low-Pass Filter
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
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 100 \
    --cutoff_ratio "$cutoff_ratio"
done
