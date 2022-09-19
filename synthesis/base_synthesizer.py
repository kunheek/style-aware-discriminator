import torch
from torchvision import transforms
from torchvision.utils import make_grid

import data
from mylib import torch_utils


class BaseSynthesizer:

    @staticmethod
    def add_commandline_args(parser):
        return parser

    def __init__(self, run_dir, image_size, folders, **kwargs):
        self.run_dir = run_dir
        self.image_size = image_size
        self.folders = folders
        self._is_available = False
        self.prepare_synthesis()

    def prepare_synthesis(self):
        raise NotImplementedError

    def is_available(self):
        return self._is_available

    def synthesize(self, model, *args, **kwargs):
        raise NotImplementedError

    def get_dataset(self, folder, force_square=True, seed=0, repeat=False):
        target_size = (self.image_size,)*2 if force_square else self.image_size
        interpolation = transforms.InterpolationMode.BICUBIC
        mean = std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        dataset = data.build_dataset(folder, transform, seed, repeat)
        return dataset

    def save_image(self, image, fname, unnormalize=True, nrow=None):
        assert isinstance(image, torch.Tensor) and image.dim() in (3, 4)
        if image.dim() == 4:
            if image.size(1) == 1:
                image = image.squeeze(0)
            else:
                if nrow is None:
                    nrow = image.size(0)
                image = make_grid(image, nrow=nrow, padding=0)

        if unnormalize:
            image = torch_utils.unnormalize(image, to_uint8=True)
        else:
            image = image.mul(255.0).add(0.5).clamp(0, 255)
            image = image.to(torch.uint8)
        pil_image = transforms.functional.to_pil_image(image)
        pil_image.save(fname)
