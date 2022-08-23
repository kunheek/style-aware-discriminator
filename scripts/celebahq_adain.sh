#!/bin/bash
DATADIR="../datasets/celeba_hq"
python train.py --mod-type adain \
    --total-nimg 1.6M --batch-size 16 --image-size 256 --load-size 320 --crop-size 256 \
    --train-dataset $DATADIR/train --eval-dataset $DATADIR/val \
    --nb-proto 64 --lambda-r1 2.0 --lambda-rec 0.3 --cutout true --color-distort true \
    --extra-desc adain
