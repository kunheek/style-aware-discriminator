import os
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.datasets import ImageFolder, LSUNClass

from mylib import misc


def make_dataset(dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), str(dir) + " is not a valid directory."
    assert max_dataset_size > 0

    samples = []
    label, target = None, 0
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if misc.is_image_file(fname):
                if label is None:
                    label = root
                elif root != label:
                    label = root
                    target += 1
                path = os.path.join(root, fname)
                samples.append((path, target))
    if not samples:
        raise RuntimeError(
            f"Found 0 images in: {root}\n"
            "Supported image extensions are: " + ",".join(misc.IMG_EXTENSIONS)
        )
    return samples[:min(max_dataset_size, len(samples))]


class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, return_target=False):
        self.samples = make_dataset(root)
        self.classes = set([y for _, y in self.samples])
        self.transform = transform
        self.return_target = return_target

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return (image, target) if self.return_target else image


class LMDBDataset(LSUNClass):
    def __init__(self, root, transform=None):
        super().__init__(root, transform)

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image #, 0


class DataPipe(torch.utils.data.IterableDataset):
    def __init__(self, dataset, seed=0, shuffle=True, drop_last=True):
        assert isinstance(dataset, (Sequence, torch.utils.data.Dataset))
        self.data = dataset
        self.seed = seed
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = 1
            rank = 0

        mod = world_size
        shift = rank
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            mod *= worker_info.num_workers
            shift = shift * worker_info.num_workers + worker_info.id

        epoch = 0
        keys = np.arange(len(self.data))
        remainder = len(keys) % mod

        while True:
            if self.shuffle:
                rng = np.random.default_rng(seed=self.seed + epoch)
                rng.shuffle(keys)

            if remainder == 0:
                indices = keys
            elif self.drop_last:
                indices = keys[:-remainder]
            else:
                indices = np.concatenate((keys, keys[:mod-remainder]))

            for index in indices[shift::mod]:
                yield self.data[index]

            epoch += 1


def build_dataset(root, transform, seed=None, repeat=False):
    if "lsun" in root:
        dataset = LMDBDataset(root, transform)
    elif os.path.isdir(root):
        dataset = ImageFolder(root, transform)
    else:
        raise NotImplementedError(root)
    if repeat:
        dataset = DataPipe(dataset, seed)
    return dataset
