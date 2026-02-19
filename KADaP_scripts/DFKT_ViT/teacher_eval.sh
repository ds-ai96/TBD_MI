gpu=$1

for model in deit_tiny_16_cifar100 deit_base_16_cifar100 deit_tiny_16_cifar10 deit_base_16_cifar10; do
    python main_kt.py \
        --mode DMI \
        --model "$model" \
        --gpu "$gpu" \
        --project_name "KADaP_L40_Teacher_Eval" \
        --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/ \
        --model_path /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/models/ \
        --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/DFKT_save_img \
        --wandb \
        --eval_teacher
done
    
