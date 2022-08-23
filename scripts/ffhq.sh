#!/bin/bash
DATADIR="../datasets/ffhq/images1024x1024"
args="--total-nimg 25M --batch-size 64 --image-size 256 --load-size 320 --crop-size 256
--train-dataset $DATADIR --eval-dataset $DATADIR
--nb-proto 128 --lambda-r1 2.0 --latent-dim 512 --latent-ratio 0.5 --cutout true --color-distort true
--extra-desc 25M"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py $(echo $args|tr -d '\r')
