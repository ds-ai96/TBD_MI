gpu=$1
mode=$2

for it in $(seq 100 100 4000); do
python data_generation.py \
    --gpu "$gpu" \
    --mode "$mode" \
    --dataset /home/mjatwk/data/imagenet/ \
    --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
    --iterations "$it"
done
