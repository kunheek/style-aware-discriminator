import torch
import torch.nn as nn
from torchvision import transforms

from .functional import pad_random_affine


class PadRandomAffine(nn.Module):
    def __init__(self, angle=0, translate=(0, 0), scale=1.0, shear=(0,0)):
        super().__init__()
        assert isinstance(angle, (int, float))
        assert isinstance(translate, (tuple, list))
        assert isinstance(scale, (int, float, tuple, list))
        assert isinstance(shear, (tuple, list))
        self.angle = float(angle)
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.interpolation = transforms.InterpolationMode.BILINEAR

    def forward(self, img):
        return pad_random_affine(
            img, self.angle, self.translate,
            self.scale, self.shear, self.interpolation,
        )


class Resize(nn.ModuleDict):
    def __init__(self, load_size, crop_size, p_crop=0.5):
        interpolation = transforms.InterpolationMode.BICUBIC
        self.p_crop = p_crop
        super().__init__({
            "load": transforms.Resize(load_size, interpolation),
            "centercrop": transforms.CenterCrop((crop_size,)*2),
            "randcrop": transforms.RandomCrop((crop_size,)*2),
            "resize": transforms.Resize((crop_size,)*2, interpolation),
        })

    def forward(self, image):
        w, h = image.size
        image = self["load"](image)
        if torch.rand(1) < self.p_crop:
            image = self["randcrop"](image)
        elif w != h:
            image = self["centercrop"](image)
        else:
            image = self["resize"](image)
        return image
