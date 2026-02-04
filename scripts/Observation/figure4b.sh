gpu=$1

python observation_python/fig4_b.py \
    --gpu "$gpu" \
    --imagenet_root /home/mjatwk/data/imagenet/ \
    --dmi_folder ./observation/08_Confidence/DMI \
    --smi_folder ./observation/08_Confidence/SMI \
    --output_dir ./observation/00_Figures \
    --num_bins 10