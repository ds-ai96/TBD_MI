gpu=$1

# for num_images in 128 256 512 1024 2048 4096 5120; do
# 원래 kr-batchsize 128 / synthetic_bs는 설정 x (128 미만인 경우만 설정)
for num_images in 8 16 32 64; do
    python main_kt_off.py \
        --mode SMI \
        --model deit_tiny_16_cifar10 \
        --gpu "$gpu" \
        --project_name "KADaP_L40_DFKT_Num_Images" \
        --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/ \
        --model_path /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/models/ \
        --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/DFKT_save_img \
        --kr-batchsize "$num_images" \
        --total_images "$num_images" \
        --synthetic_bs "$num_images" \
        --epoches 1000 \
        --wandb
done
