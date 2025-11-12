gpu=$1

for it in $(seq 400 100 4000); do
python main_quant.py \
    --gpu "$gpu" \
    --mode SMI \
    --wandb \
    --project_name "SMI_Quant" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations "$it" \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8
done
