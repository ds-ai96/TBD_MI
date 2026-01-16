gpu=$1

for var in $(seq 0.0 0.1 5.0); do
python main_quant.py \
    --gpu "$gpu" \
    --mode DMI \
    --wandb \
    --project_name "DMI_Quant_Gaussian_New" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --variance "$var" \
    --w_bit 4 \
    --a_bit 8
done
