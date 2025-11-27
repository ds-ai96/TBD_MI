gpu=$1
nums=(1 2 4 8 16 32 64 128 160)
for num in ${nums[@]}; do
python main_quant.py \
    --gpu "$gpu" \
    --mode DMI \
    --wandb \
    --project_name "DMI_NUM" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --w_bit 4 \
    --a_bit 8 \
    --num_runs "$num"
done
