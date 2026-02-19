for w_bit in 4 8; do
    for cutoff_ratio in 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95; do
        for lpf_every in 50 100 200 300 400 500; do
            python main_quant.py \
                --mode TBD_MI \
                --model deit_tiny_16_imagenet \
                --wandb \
                --project_name "KADaP_Hyper_Idea1" \
                --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/imagenet \
                --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/save_img_hyper \
                --w_bit "$w_bit" \
                --lpf \
                --lpf_every "$lpf_every" \
                --cutoff_ratio "$cutoff_ratio"
        done
    done
done