gpu=$1

# CIFAR-10
python main_quant_cnn.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --model "resnet50_imagenet" \
    --project_name "TBDMI_CNN_TEST" \
    --dataset /home/jener05458/data/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset_quant_cnn/ \
    --iterations 2000 \
    --synthetic_bs 256 \
    --quant_mode "ptq" \
    --w_bit 8 \
    --a_bit 8
    # --dataset /home/mjatwk/data/imagenet/ \