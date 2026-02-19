for w_bit in 4 8; do
    for lambda in 0.0 0.5 1.0 1.5 2.0 2.5 3.0 4.0 4.5 5.0; do
        for sc_every in 50 100; do
        python main_quant.py \
            --mode TBD_MI \
            --model deit_tiny_16_imagenet \
            --wandb \
            --project_name "KADaP_Hyper_Idea2" \
            --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/imagenet \
            --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/save_img_hyper \
            --prune_it 50 100 200 300 \
            --prune_ratio 0.3 0.3 0.3 0.3 \
            --w_bit "$w_bit" \
            --sc_center \
            --sc_every "$sc_every" \
            --sc_center_lambda "$lambda"
        done
    done
done
