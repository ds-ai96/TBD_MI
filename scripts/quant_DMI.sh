gpu=$1

for it in $(seq 400 100 4000); do
python main_quant.py \
    --gpu "$gpu" \
    --mode DMI \
    --wandb \
    --project_name "DMI_Quant" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations "$it" \
    --w_bit 4 \
    --a_bit 8
done
