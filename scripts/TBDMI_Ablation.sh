gpu=$1

# Idea 1: Low-pass filter
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7

# Idea 1: Low-pass filter (SMI)
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7

# Idea 2: Saliency map centering + sparsification
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda 4.0

# Idea 3: Soft label
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8 \
    --use_soft_label \
    --soft_label_alpha 0.7

# Idea 3: Soft label (SMI)
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --use_soft_label \
    --soft_label_alpha 0.7

# Idea 1 + 2
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda 4.0

# Idea 1 + 3
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7 \
    --use_soft_label \
    --soft_label_alpha 0.7

# Idea 1 + 3 (SMI)
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7 \
    --use_soft_label \
    --soft_label_alpha 0.7

# Idea 2 + 3
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda 4.0 \
    --use_soft_label \
    --soft_label_alpha 0.7

# Idea 1 + 2 + 3
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "Ablation_TEST" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_every 50 \
    --cutoff_ratio 0.7 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda 4.0 \
    --use_soft_label \
    --soft_label_alpha 0.7


# TEST
# python main_quant.py \
#     --gpu "$gpu" \
#     --mode TBD_MI \
#     --wandb \
#     --project_name "Ablation_TEST" \
#     --dataset /home/mjatwk/data/imagenet/ \
#     --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
#     --iterations 4000 \
#     --prune_it 50 100 200 300 \
#     --prune_ratio 0.3 0.3 0.3 0.3 \
#     --w_bit 4 \
#     --a_bit 8 \
#     --lpf \
#     --lpf_every 50 \
#     --cutoff_ratio 0.7