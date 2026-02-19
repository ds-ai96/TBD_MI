gpu=$1

# DeepInversion
for model in deit_base_16_imagenet deit_tiny_16_imagenet; do
    python main_quant.py \
        --gpu "$gpu" \
        --mode DMI \
        --wandb \
        --project_name "KADaP_L40_Teacher_Eval" \
        --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/imagenet \
        --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/save_img \
        --iterations 4000 \
        --model "$model" \
        --eval_teacher
done