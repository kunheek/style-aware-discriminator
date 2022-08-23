#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

import data
from model import StyleAwareDiscriminator
from model.augmentation import SimpleTransform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-dataset", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--title")
    parser.add_argument("--labels", nargs="+", default=[])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint)
    assert os.path.isdir(args.target_dataset)
    return args


def plot_tsne(model, dataset, seed, filename, title, labels, batch_size):
    dataset.return_target = True
    device = model.device
    loader = torch.utils.data.DataLoader(dataset, batch_size)

    style_codes, targets = [], []
    for batch in loader:
        image, target = batch
        image = image.to(device)

        style_code = model.D_ema(image, command="encode")

        style_codes.append(style_code.cpu().numpy())
        targets.append(target.numpy())

    style_codes = np.concatenate(style_codes)
    targets = np.concatenate(targets)
    legends = len(np.unique(targets)) == len(labels)

    prototypes = model.prototypes_ema[0].weight.cpu().numpy()
    proto_target = np.max(targets) + 1
    proto_targets = np.zeros((prototypes.shape[0],), dtype=np.int32)
    proto_targets += proto_target

    features = np.vstack((style_codes, prototypes))
    targets = np.concatenate((targets, proto_targets))
    if legends:
        labels.append("prototype")
    else:
        labels = np.unique(targets)
    print(features.shape, targets.shape, labels)

    tsne = TSNE(
        n_components=2,
        learning_rate="auto",
        random_state=seed,
        init="pca",
    )
    z = tsne.fit_transform(features)

    plt.subplots(figsize=(10, 8))
    colors = "salmon", "cornflowerblue", "seagreen"
    for i, l in enumerate(labels):
        if l in (proto_target, "prototype"):
            c, m, a, ec = "k", "X", 1.0, None
        else:
            c, m, a, ec = colors[i], "o", 0.8, "w"

        x, y = z[np.where(targets==i)].T
        plt.scatter(x, y, 100, c=c, marker=m, alpha=a, edgecolors=ec, label=l)

    plt.axis("off")
    if legends:
        plt.legend()
    if title is not None:
        plt.title(title)
    plt.savefig(filename)
    plt.close("all")


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opts = checkpoint["options"]
    print(opts)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    model = StyleAwareDiscriminator(opts)
    model.load(checkpoint)
    del model.optimizers

    transform = SimpleTransform(opts.image_size)
    target_dataset = data.get_dataset(args.target_dataset, transform)

    os.makedirs(opts.run_dir, exist_ok=True)
    filename = os.path.join(opts.run_dir, "tsne.png")
    plot_tsne(
        model=model,
        dataset=target_dataset,
        seed=args.seed,
        filename=filename,
        title=args.title,
        labels=args.labels,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
