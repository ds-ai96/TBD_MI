gpu=$1

for it in $(seq 100 100 4000); do
python data_generation.py \
    --gpu "$gpu" \
    --mode SMI \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations "$it" \
    --prune_it 50 100 200 300 \
    --prune_ratio 0.3 0.3 0.3 0.3
done
