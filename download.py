#!/usr/bin/env python
import argparse
import os
import shutil
import zipfile

from huggingface_hub import hf_hub_download

FILES = {
    "stats": "stats.zip",
    "checkpoints": "checkpoints.zip",
    "afhq-adain": "afhq-adain.pt",
    "afhq-stylegan2": "afhq-stylegan2-5M.pt",
    "afhqv2": "afhqv2-512x512-5M.pt",
    "celebqhq-adain": "celebqhq-adain.pt",
    "celebqhq-stylegan2": "celebqhq-stylegan2-5M.pt",
    "church": "church-25M.pt",
    "ffhq": "ffhq-25M.pt",
    "flower": "flower-256x256-adain.pt",
}


def main():
    parser = argparse.ArgumentParser(description="Download a file from the Hugging Face Hub.")
    parser.add_argument("files", type=str, nargs="+", help="Files to download")
    args = parser.parse_args()

    if args.files == ["all"]:
        files_to_download = ["stats", "checkpoints"]
    else:
        files_to_download = args.files

    for file in files_to_download:
        if file not in FILES:
            raise ValueError(f"Unknown file '{file}'")

        if FILES[file].endswith(".zip"):
            local_dir = "tmp"
            os.makedirs(local_dir, exist_ok=True)
        elif file == "stats":
            local_dir = "assets"
        else:
            local_dir = "checkpoints"

        hf_hub_download(
            repo_id="kunheekim/style-aware-discriminator",
            filename=FILES[file],
            local_dir=local_dir,
        )

        if FILES[file].endswith(".zip"):
            full_path = os.path.join(local_dir, FILES[file])
            if file == "stats":
                target_path = "./assets"
            else:
                target_path = "./checkpoints"
            with zipfile.ZipFile(full_path, "r") as zip_ref:
                zip_ref.extractall(target_path)
            shutil.rmtree(local_dir)


if __name__ == "__main__":
    main()
