#!/bin/bash

# Make sure 'wget' and 'unzip' is installed.
command -v wget >/dev/null 2>&1 || {
    echo "require 'wget' but it's not installed. Aborting." >&2; exit 1;
}
command -v unzip >/dev/null 2>&1 || {
    echo "require 'unzip' but it's not installed. Aborting." >&2; exit 1;
}

FILE=$1

if [ $FILE == "all" ]; then
    URL=https://www.dropbox.com/s/vxnpfut9wgwe7js/stats.zip?dl=0
    ZIP_FILE=./assets/stats.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./assets
    rm $ZIP_FILE

    URL=https://www.dropbox.com/s/rodbv3zuq77864i/checkpoints.zip?dl=0
    ZIP_FILE=./checkpoints.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d .
    rm $ZIP_FILE

elif [ $FILE == "stats" ]; then
    URL=https://www.dropbox.com/s/vxnpfut9wgwe7js/stats.zip?dl=0
    ZIP_FILE=./assets/stats.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./assets
    rm $ZIP_FILE

elif [ $FILE == "checkpoints" ]; then
    URL=https://www.dropbox.com/s/rodbv3zuq77864i/checkpoints.zip?dl=0
    ZIP_FILE=./checkpoints.zip
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d .
    rm $ZIP_FILE

elif  [ $FILE == "afhq-adain" ]; then
    URL=https://www.dropbox.com/s/ph3rc6fcf8mpnl0/afhq-adain.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/afhq-adain.pt
    wget -N $URL -O $OUT_FILE
elif  [ $FILE == "afhq-stylegan2" ]; then
    URL=https://www.dropbox.com/s/esc04uy5fopew1i/afhq-stylegan2-5M.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/afhq-stylegan2-5M.pt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "afhqv2" ]; then
    URL=https://www.dropbox.com/s/ro1dqyenlrgkq4x/afhqv2-512x512-5M.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/afhqv2-512x512-5M.pt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "celebahq-adain" ]; then
    URL=https://www.dropbox.com/s/wgph4klpb0rweng/celebahq-adain.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/celebahq-adain.pt
    wget -N $URL -O $OUT_FILE
elif  [ $FILE == "celebahq-stylegan2" ]; then
    URL=https://www.dropbox.com/s/ndywhps4845fmga/celebahq-stylegan2-5M.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/celebahq-stylegan2-5M.pt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "church" ]; then
    URL=https://www.dropbox.com/s/5o3t4q84u5ozoch/church-25M.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/church-25M.pt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "ffhq" ]; then
    URL=https://www.dropbox.com/s/rqov8x82plr98jx/ffhq-25M.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/ffhq-25M.pt
    wget -N $URL -O $OUT_FILE

elif  [ $FILE == "flower" ]; then
    URL=https://www.dropbox.com/s/pqhyj9t69t0esw2/flower-256x256-adain.pt?dl=0
    mkdir -p ./checkpoints
    OUT_FILE=./checkpoints/flower-256x256-adain.pt
    wget -N $URL -O $OUT_FILE

else
    echo "Unsupported arguments. Available arguments are:"
    echo "  all, stats, checkpoints,"
    echo "  afhq-adain, afhq-stylegan2, celebahq-adain, celebahq-stylegan2, afhqv2, church, ffhq, and flower"
    exit 1

fi