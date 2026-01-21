gpu=$1

# synthetic_bs 원래 32
python main_quant_cnn.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --model "resnet20_cifar10" \
    --project_name "TBDMI_CNN_TEST" \
    --dataset /home/jener05458/data/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset_quant_cnn/ \
    --iterations 4000 \
    --synthetic_bs 32 \
    --quant_mode "qat" \
    --w_bit 4 \
    --a_bit 4 \
    --temperature 4 \
    --alpha 10 \
    --lambda_ce 1.0 \
