gpu=$1

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
    --cutoff_ratio 0.7 \
    --sc_center \
    --sc_every 100 \
    --sc_center_lambda 0.01 \
    --use_soft_label \
    --soft_label_alpha 0.6
