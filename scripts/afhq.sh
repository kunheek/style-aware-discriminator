#!/bin/bash
DATADIR="../datasets/afhq"
python train.py \
    --total-nimg 5M --batch-size 32 --image-size 256 --load-size 320 --crop-size 256 \
    --train-dataset $DATADIR/train --eval-dataset $DATADIR/val \
    --nb-proto 32 --lambda-r1 1.0 \
    --extra-desc stylegan2 5M
