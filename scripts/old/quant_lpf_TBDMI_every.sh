gpu=$1

for every in $(seq 50 50 4000); do
python main_quant.py \
    --gpu "$gpu" \
    --mode TBD_MI \
    --wandb \
    --project_name "TBD_MI_Quant_LPF_Search_Every" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations 4000 \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3 \
    --w_bit 4 \
    --a_bit 8 \
    --lpf \
    --lpf_start 0 \
    --lpf_every "$every" \
    --cutoff_ratio 0.8 \
    --sc_center \
    --sc_warmup 0 \
    --sc_every 100 \
    --sc_center_lambda 1.0
done
