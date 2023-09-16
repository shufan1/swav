#!/bin/bash

DATASET_PATH="/scratch/sl636/ILSVRC/Data/CLS-LOC/"
DUMP_PATH="experiments/imagenet_classification_scratch400ep_mycode"
MODEL_PATH="/shared/data/sx78/swav_imagenet_weigths/imagenet_from_scratch_400ep.pt"
mkdir -p $DUMP_PATH

python parallel_train_scratch.py \
--total_epochs 10 \
--save_every 10 \
--batch_size 32 \
--dump_path $DUMP_PATH \
--data_path $DATASET_PATH \
--model_path $MODEL_PATH \
--lr 0.3 \
--wd 1e-6 \