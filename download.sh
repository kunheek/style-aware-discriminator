#!/bin/bash

# Make sure 'wget' and 'unzip' are installed.
command -v wget >/dev/null 2>&1 || {
    echo "require 'wget' but it's not installed. Aborting." >&2; exit 1;
}
command -v unzip >/dev/null 2>&1 || {
    echo "require 'unzip' but it's not installed. Aborting." >&2; exit 1;
}

FILE=$1

if [ $FILE == "all" ]; then
    wget -O stats.zip "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EcoST7aF9ppJvTEAK8135p8BZEDzz42Q4J9hTfSf99x6rA?e=cklSbP&download=1"
    unzip stats.zip -d assets
    rm stats.zip

    wget -O checkpoints.zip "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EX0kn7Wq86BBsAXSu1SYZw4BOhQOVQ8Jua_jBSo8eDyvsQ?e=vnmkH7&download=1"
    unzip checkpoints.zip
    rm checkpoints.zip

elif [ $FILE == "stats" ]; then
    wget -O stats.zip "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EcoST7aF9ppJvTEAK8135p8BZEDzz42Q4J9hTfSf99x6rA?e=cklSbP&download=1"
    unzip stats.zip -d assets
    rm stats.zip

elif [ $FILE == "checkpoints" ]; then
    wget -O checkpoints.zip "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EX0kn7Wq86BBsAXSu1SYZw4BOhQOVQ8Jua_jBSo8eDyvsQ?e=vnmkH7&download=1"
    unzip checkpoints.zip
    rm checkpoints.zip

elif  [ $FILE == "afhq-adain" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhq-adain.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EXXn2WmxRTNMuHwEip_7ISYBZP0yjIGLEdrnXRpWq_4_vw?e=Q79XI7&download=1"

elif  [ $FILE == "afhq-stylegan2" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhq-stylegan2-5M.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EYFJoQILF8ZPvVla5CUMzzcBKwFJKOnOYmS92VxE--0PQg?e=qqKov4&download=1"

elif  [ $FILE == "afhqv2" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhqv2-512x512-5M.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EQZWO4260J9DlRqJvGNLZ8gBJ_yzzPlXjiWFcDbxMdSIAA?e=qrmtaC&download=1"

elif  [ $FILE == "celebahq-adain" ]; then
    mkdir checkpoints
    wget -O checkpoints/celebahq-adain.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/Ef6XeyapXGdDgxNz36PDYBwBgOdZ4KfZq3lSd1Ak0qAOKA?e=ORFas5&download=1"

elif  [ $FILE == "celebahq-stylegan2" ]; then
    mkdir checkpoints
    wget -O checkpoints/celebahq-stylegan2-5M.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EXmytLCR6-ZMkzcROcRmQk8Bcb4lbSMVmspydMV4CjNJuQ?e=0CrzKJ&download=1"

elif  [ $FILE == "church" ]; then
    mkdir checkpoints
    wget -O checkpoints/church-25M.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EagzHup1K2ZFsKVzwHswg4cBxkYphYVkxZjAtMirFqfb4Q?e=Fi82ap&download=1"

elif  [ $FILE == "ffhq" ]; then
    mkdir checkpionts
    wget -O checkpoints/ffhq-25M.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EUREXa2eJlpLgbEUyr5usFoB5XAKGx_5enx42ozIeJoatQ?e=RoRRL4&download=1"

elif  [ $FILE == "flower" ]; then
    mkdir checkpionts
    wget -O checkpionts/flower-256x256-adain.pt "https://o365kaist-my.sharepoint.com/:u:/g/personal/kunheekim_office_kaist_ac_kr/EWc9bn6MiSJAh_yzQxVvwUIBN8rckJ0mi_IO1BuIwYqmKA?e=yK87ZZ&download=1"

else
    echo "Unsupported arguments. Available arguments are:"
    echo "  all, stats, checkpoints,"
    echo "  afhq-adain, afhq-stylegan2, celebahq-adain, celebahq-stylegan2, afhqv2, church, ffhq, and flower"
fi
