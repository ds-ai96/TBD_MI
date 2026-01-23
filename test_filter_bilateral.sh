gpu=$1

for kernel_size in 5 7; do
    for sigma_s in 1.5 2.0 2.5 3.0; do
        for sigma_r in 0.05 0.10 0.15 0.20; do
            python main_quant.py \
                --gpu "$gpu" \
                --mode TBD_MI \
                --wandb \
                --project_name "Idea1_Bilateral_Filter" \
                --dataset /home/mjatwk/data/imagenet/ \
                --datapool /home/jener05458/src/EdgeMI/TBD_MI/dataset/ \
                --iterations 4000 \
                --w_bit 4 \
                --a_bit 8 \
                --lpf \
                --lpf_type "bilateral" \
                --lpf_every 100 \
                --bi_kernel "$kernel_size" \
                --bi_sigma_s "$sigma_s" \
                --bi_sigma_r "$sigma_r"
        done
    done
done
