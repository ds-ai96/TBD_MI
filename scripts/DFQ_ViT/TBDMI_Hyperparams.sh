
gpu=$1

# Idea 1 + 2 + 3
for w_bit in 4 8; do
    for cutoff_ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
        for sc_center_lambda in 0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0; do
            for soft_label_alpha in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                python main_quant.py \
                    --gpu "$gpu" \
                    --mode TBD_MI \
                    --wandb \
                    --project_name "DFQ_VIT_Hyperparameter_Search" \
                    --dataset /home/mjatwk/data/imagenet/ \
                    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
                    --iterations 4000 \
                    --prune_it 50 100 200 300 \
                    --prune_ratio 0.3 0.3 0.3 0.3 \
                    --w_bit "$w_bit" \
                    --a_bit 8 \
                    --lpf \
                    --lpf_type "gaussian" \
                    --lpf_every 50 \
                    --cutoff_ratio "$cutoff_ratio" \
                    --sc_center \
                    --sc_every 100 \
                    --sc_center_lambda "$sc_center_lambda" \
                    --use_soft_label \
                    --soft_label_alpha "$soft_label_alpha"
            done
        done
    done
done
