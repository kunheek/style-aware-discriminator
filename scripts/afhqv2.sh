#!/bin/bash
DATADIR="../datasets/afhq_v2"
args="--total-nimg 5M --batch-size 32 --image-size 512
--train-dataset $DATADIR/train --eval-dataset $DATADIR/val
--nb-proto 32 --lambda-r1 2.0 --cnt-res 32
--extra-desc 5M"
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py $(echo $args|tr -d '\r')
