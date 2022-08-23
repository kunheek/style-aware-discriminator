#!/bin/bash
DATADIR="../datasets/oxford-102"
python train.py \
    --total-nimg 1.6M --batch-size 16 --image-size 256 --load-size 320 --crop-size 256 \
    --train-dataset $DATADIR/train --eval-dataset $DATADIR/valid \
    --nb-proto 128 --lambda-r1 10.0
