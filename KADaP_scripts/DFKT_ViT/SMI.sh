gpu=$1

# for model in deit_tiny_16_cifar10 deit_base_16_cifar10 deit_tiny_16_cifar100 deit_base_16_cifar100; do
python main_kt.py \
    --mode SMI \
    --model deit_tiny_16_cifar100 \
    --gpu "$gpu" \
    --project_name "KADaP_L40_DFKT_Baseline_120" \
    --dataset /root/kadap/MyDisk/tools/ide/jhchoi/data/ \
    --model_path /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/models/ \
    --datapool /root/kadap/MyDisk/tools/ide/jhchoi/TBD_MI/DFKT_save_img \
    --kr-batchsize 128 \
    --epoches 1000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --wandb
# done
    
