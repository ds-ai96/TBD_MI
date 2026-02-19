gpu=$1

# DeepInversion
for w_bit in 4 8; do
    for model in deit_base_16_imagenet deit_tiny_16_imagenet; do
        python main_quant.py \
            --gpu "$gpu" \
            --mode SMI \
            --wandb \
            --project_name "KADaP_H100_Baseline" \
            --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/imagenet \
            --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/save_img \
            --iterations 4000 \
            --prune_it 50 100 200 300 \
            --prune_ratio 0.3 0.3 0.3 0.3 \
            --w_bit "$w_bit" \
            --a_bit 8 \
            --model "$model"
    done
done
