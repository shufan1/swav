#!/bin/bash

MODEL_PATH="/home/sx78/SSRS/swav_fb/swav/experiments/swav_10ep_pretrain_wandb/checkpoints/ckp-9.pth"
OUTPUT_PATH="/home/sx78/SSRS/swav_fb/swav/experiments/swav_10ep_pretrain_wandb/checkpoints/climatenet_subset_epoch9.pth"
python save_model.py --model_path $MODEL_PATH --output_path $OUTPUT_PATH 
