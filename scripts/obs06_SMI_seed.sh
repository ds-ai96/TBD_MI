gpu=$1

for seed in $(seq 0 1 50); do
python main_quant.py \
    --gpu "$gpu" \
    --mode SMI_SEED \
    --wandb \
    --project_name "SMI_SEED" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --seed "$seed"
done
