gpu=$1

for seed in $(seq 0 1 50); do
python main_quant.py \
    --gpu "$gpu" \
    --mode DMI \
    --wandb \
    --project_name "DMI_SEED" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8 \
    --seed "$seed"
done
