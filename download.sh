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
    wget -O stats.zip "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/ETssKsRsOA5HlxJlP2mlL24Bdjp1zhpWGjtzmmD1TkWmgg?e=gN55yw&download=1"
    unzip stats.zip -d assets
    rm stats.zip

    wget -O checkpoints.zip "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EVWyUG_advFLiaUbSOVSdMgBTleTw1hufu1583aCnuBK_w?e=eNF1q0&download=1"
    unzip checkpoints.zip
    rm checkpoints.zip

elif [ $FILE == "stats" ]; then
    wget -O stats.zip "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/ETssKsRsOA5HlxJlP2mlL24Bdjp1zhpWGjtzmmD1TkWmgg?e=gN55yw&download=1"
    unzip stats.zip -d assets
    rm stats.zip

elif [ $FILE == "checkpoints" ]; then
    wget -O checkpoints.zip "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EVWyUG_advFLiaUbSOVSdMgBTleTw1hufu1583aCnuBK_w?e=eNF1q0&download=1"
    unzip checkpoints.zip
    rm checkpoints.zip

elif  [ $FILE == "afhq-adain" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhq-adain.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/Ecd6a64y6ZVCnpdLXB7Fb4sBiCpG48m2ncxAKR_aJpLWzA?e=1NTn3r&download=1"

elif  [ $FILE == "afhq-stylegan2" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhq-stylegan2-5M.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EX0JEvsGdQtJi-SeldrsMfUBngby8yxx_76pX2c8wtQidA?e=TmIRkC&download=1"

elif  [ $FILE == "afhqv2" ]; then
    mkdir checkpoints
    wget -O checkpoints/afhqv2-512x512-5M.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/Ec8tWDfZtK1Dmm_bw8azbi8BSfy_MUJkv8fbkE4_XxuhmQ?e=tNiGQp&download=1"

elif  [ $FILE == "celebahq-adain" ]; then
    mkdir checkpoints
    wget -O checkpoints/celebahq-adain.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EYOufvKpdMRDkavr2OfZbZwBiVN0XoksrtXEnn5krPh7UQ?e=7b3kzG&download=1"

elif  [ $FILE == "celebahq-stylegan2" ]; then
    mkdir checkpoints
    wget -O checkpoints/celebahq-stylegan2-5M.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EU8HgMXm9dNBj31LUIdc90kBQqEAJmmfkoAi4-BGRrzYAA?e=935Zrs&download=1"

elif  [ $FILE == "church" ]; then
    mkdir checkpoints
    wget -O checkpoints/church-25M.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/Efn0z0mbDLZBobxByA6mKEEB-27zRfEprUt62-567oLWiQ?e=ZLM7YD&download=1"

elif  [ $FILE == "ffhq" ]; then
    mkdir checkpionts
    wget -O checkpoints/ffhq-25M.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EfbsIjHka1JGsYHjRE2KrVEBwSjJ11ThbwKm5TVIpvUiAA?e=TgKMcP&download=1"

elif  [ $FILE == "flower" ]; then
    mkdir checkpionts
    wget -O checkpionts/flower-256x256-adain.pt "https://postechackr-my.sharepoint.com/:u:/g/personal/kunkim_postech_ac_kr/EeiK2iNEGMFPmYIsosF_D9ABd09_wN4MF3KCauLAQxgH0g?e=Jy5MJt&download=1"

else
    echo "Unsupported arguments. Available arguments are:"
    echo "  all, stats, checkpoints,"
    echo "  afhq-adain, afhq-stylegan2, celebahq-adain, celebahq-stylegan2, afhqv2, church, ffhq, and flower"
fi
