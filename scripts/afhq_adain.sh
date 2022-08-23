#!/bin/bash
DATADIR="../datasets/afhq"
python train.py --mod-type adain \
    --total-nimg 1.6M --batch-size 16 --image-size 256 --load-size 320 --crop-size 256 \
    --train-dataset $DATADIR/train --eval-dataset $DATADIR/val \
    --nb-proto 32 --lambda-rec 0.3 \
    --extra-desc adain
