#!/bin/bash

# Make sure 'wget' and 'unzip' are installed.
command -v wget >/dev/null 2>&1 || {
    echo "require 'wget' but it's not installed. Aborting." >&2; exit 1;
}
command -v unzip >/dev/null 2>&1 || {
    echo "require 'unzip' but it's not installed. Aborting." >&2; exit 1;
}

get_filename_from_url () {
    local URL=$1
    local arrURL=(${URL//// })  # {IN//'split symbol'/ }
    local filename=${arrURL[-1]}
    filename=$(echo "$filename" | sed -e "s/"?dl=0"//")
    echo "$filename"
}

download_dropbox_url () {
    local URL=$1
    local DIR=$2

    local filename=$(get_filename_from_url $URL)
    local out_file=$DIR/$filename
    mkdir -p $2
    wget -N $URL -O $out_file
    echo "$out_file"
}

download_unzip_dropbox_url () {
    local URL=$1
    local DIR1=$2  # download directory.
    local DIR2=$3  # unzip directory.

    local filename=$(download_dropbox_url $URL $DIR1)
    unzip $filename -d $DIR2
    rm $filename
}


FILE=$1

if [ $FILE == "all" ]; then
    URL=https://www.dropbox.com/s/vxnpfut9wgwe7js/stats.zip?dl=0
    download_unzip_dropbox_url $URL . assets

    URL=https://www.dropbox.com/s/rodbv3zuq77864i/checkpoints.zip?dl=0
    download_unzip_dropbox_url $URL . .

elif [ $FILE == "stats" ]; then
    URL=https://www.dropbox.com/s/vxnpfut9wgwe7js/stats.zip?dl=0
    download_unzip_dropbox_url $URL . assets

elif [ $FILE == "checkpoints" ]; then
    URL=https://www.dropbox.com/s/rodbv3zuq77864i/checkpoints.zip?dl=0
    download_unzip_dropbox_url $URL . .

elif  [ $FILE == "afhq-adain" ]; then
    URL=https://www.dropbox.com/s/ph3rc6fcf8mpnl0/afhq-adain.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "afhq-stylegan2" ]; then
    URL=https://www.dropbox.com/s/esc04uy5fopew1i/afhq-stylegan2-5M.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "afhqv2" ]; then
    URL=https://www.dropbox.com/s/ro1dqyenlrgkq4x/afhqv2-512x512-5M.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "celebahq-adain" ]; then
    URL=https://www.dropbox.com/s/wgph4klpb0rweng/celebahq-adain.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "celebahq-stylegan2" ]; then
    URL=https://www.dropbox.com/s/ndywhps4845fmga/celebahq-stylegan2-5M.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "church" ]; then
    URL=https://www.dropbox.com/s/5o3t4q84u5ozoch/church-25M.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "ffhq" ]; then
    URL=https://www.dropbox.com/s/rqov8x82plr98jx/ffhq-25M.pt?dl=0
    download_dropbox_url $URL "checkpoints"

elif  [ $FILE == "flower" ]; then
    URL=https://www.dropbox.com/s/pqhyj9t69t0esw2/flower-256x256-adain.pt?dl=0
    download_dropbox_url $URL "checkpoints"

else
    echo "Unsupported arguments. Available arguments are:"
    echo "  all, stats, checkpoints,"
    echo "  afhq-adain, afhq-stylegan2, celebahq-adain, celebahq-stylegan2, afhqv2, church, ffhq, and flower"
fi
