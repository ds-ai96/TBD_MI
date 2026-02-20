gpu=$1

# CIFAR-10
python main_quant_cnn.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --model "resnet20_cifar10" \
    --project_name "TBDMI_CNN_TEST" \
    --dataset /home/jener05458/data/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset_quant_cnn/ \
    --iterations 2000 \
    --synthetic_bs 256 \
    --quant_mode "ptq" \
    --w_bit 4 \
    --a_bit 4
