#!/usr/bin/env python
import argparse
import os
import shutil

import torch
import torch.nn.functional as F
from PIL import Image

import data
from model import StyleAwareDiscriminator
from model.augmentation import SimpleTransform
from mylib import misc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--query")
    parser.add_argument("--target-dataset")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--allow-tf32", type=misc.str2bool, default=False)
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint)
    assert os.path.isfile(args.query)
    assert os.path.isdir(args.target_dataset)
    return args


def style_similarity(query, key):
    # query and key are already normalized.
    sim = (query-key).square().sum(dim=1)
    return sim


def content_similarity(query, key, center_only=False):
    if center_only:
        query = query[:, :, 2:14, 2:14]
        key = key[:, :, 2:14, 2:14]
    query = F.normalize(query, dim=1)
    key = F.normalize(key, dim=1)
    sim = (query-key).square().sum(1)
    return sim.mean((1,2))


@torch.no_grad()
def search(model, query, target_dataset, k=5):
    print("Performing similarity search ...")
    device = model.device

    query = Image.open(query).convert("RGB")
    query = target_dataset.transform(query).unsqueeze(0).to(device)
    loader = torch.utils.data.DataLoader(target_dataset, batch_size=1)

    query_content = model.G_ema(query, command="encode")
    query_style = model.D_ema(query, command="encode")

    content_sims, style_sims = [], []
    for image in loader:
        image = image.to(device)

        key_content = model.G_ema(image, command="encode")
        key_style = model.D_ema(image, command="encode")

        content_sim = content_similarity(query_content, key_content, center_only=True)
        style_sim = style_similarity(query_style, key_style)

        content_sims.append(content_sim.cpu().view(-1))
        style_sims.append(style_sim.cpu().view(-1))

    content_sims = torch.cat(content_sims)
    style_sims = torch.cat(style_sims)

    _, content_neighbors = torch.topk(-content_sims, k=k)
    _, style_neighbors = torch.topk(-style_sims, k=k)
    return content_neighbors, style_neighbors


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opts = checkpoint["options"]
    print(opts)

    print(f"=> allow tf32 = {args.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.allow_tf32 = args.allow_tf32
    torch.backends.cudnn.benchmark = False

    model = StyleAwareDiscriminator(opts)
    model.load(checkpoint)
    del model.optimizers

    transform = SimpleTransform(opts.image_size)
    target_dataset = data.get_dataset(args.target_dataset, transform)

    content_neighbors, style_neighbors = search(
        model=model,
        query=args.query,
        target_dataset=target_dataset,
        k=args.k,
    )

    target_dataset.transform = None
    save_dir = os.path.join(opts.run_dir, "similarity_search")
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(args.query, os.path.join(save_dir, "query.jpg"))
    for i, (j, k) in enumerate(zip(content_neighbors, style_neighbors)):
        if i >= args.k:
            break
        image = target_dataset[j]
        image.save(os.path.join(save_dir, f"content_{i}.jpg"))
        image = target_dataset[k]
        image.save(os.path.join(save_dir, f"style_{i}.jpg"))


if __name__ == "__main__":
    main()
