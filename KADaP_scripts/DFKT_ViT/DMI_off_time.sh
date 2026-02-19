gpu=$1

for time in 4400 4800 5200 5600 6000; do
    python main_kt_time.py \
        --mode SMI \
        --model deit_tiny_16_cifar10 \
        --gpu "$gpu" \
        --project_name "KADaP_L40_DFKT_Time_Constraint" \
        --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/ \
        --model_path /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/models/ \
        --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/DFKT_save_img \
        --synth_time_budget "$time" \
        --epoches 1000 \
        --synthetic_bs 128 \
        --wandb
done
