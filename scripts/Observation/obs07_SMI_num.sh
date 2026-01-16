gpu=$1
nums=(1 2 4 8 16 32 64 128 160)
for num in ${nums[@]}; do
python main_quant.py \
    --gpu "$gpu" \
    --mode SMI \
    --wandb \
    --project_name "SMI_NUM" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 8 \
    --a_bit 8 \
    --num_runs "$num"
done
