#!/bin/bash
DATADIR="../datasets/lsun"
args="--total-nimg 25M --batch-size 64 --image-size 256 --load-size 320 --crop-size 256
--train-dataset $DATADIR/church_outdoor_train_lmdb --eval-dataset $DATADIR/church_outdoor_val_lmdb
--nb-proto 128 --latent-ratio 0.5
--extra-desc 25M"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py $(echo $args|tr -d '\r')
