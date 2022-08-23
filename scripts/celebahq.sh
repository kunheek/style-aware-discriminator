#!/bin/bash
DATADIR="../datasets/celeba_hq"
python train.py \
    --total-nimg 5M --batch-size 32 --image-size 256 --load-size 320 --crop-size 256 \
    --train-dataset $DATADIR/train --eval-dataset $DATADIR/val \
    --nb-proto 64 --lambda-r1 2.0 --cutout true \
    --extra-desc stylegan2 5M
