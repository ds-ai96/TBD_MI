gpu=$1
num_images=$2

python main_kt_off.py \
    --mode SMI \
    --model deit_tiny_16_cifar10 \
    --gpu "$gpu" \
    --project_name "KADaP_L40_DFKT_Baseline" \
    --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/ \
    --model_path /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/models/ \
    --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/DFKT_save_img \
    --kr-batchsize 128 \
    --total_images "$num_images" \
    --epoches 1000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --wandb    
