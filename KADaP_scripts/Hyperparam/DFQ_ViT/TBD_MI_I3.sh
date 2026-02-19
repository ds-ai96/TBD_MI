for w_bit in 4 8; do
    for alpha in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        python main_quant.py \
            --mode TBD_MI \
            --model deit_tiny_16_imagenet \
            --wandb \
            --project_name "KADaP_Hyper_Idea3" \
            --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/imagenet \
            --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/save_img_hyper \
            --w_bit "$w_bit" \
            --use_soft_label \
            --soft_label_alpha "$alpha"
    done
done
