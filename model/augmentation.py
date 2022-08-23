from torchvision import transforms

from mylib.augmentations import PadRandomAffine, Resize
from mylib.misc import str2bool

BICUBIC = transforms.InterpolationMode.BICUBIC


class Augmentation:

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--load-size", type=int, default=320)  # Used 320 for 256^2.
        parser.add_argument("--crop-size", type=int)
        parser.add_argument("--min-scale-crop", type=float, default=0.250)
        parser.add_argument("--rotation-degree", type=float, default=30.0)
        parser.add_argument("--cutout", type=str2bool, default=False)
        parser.add_argument("--color-distort", type=str2bool, default=False)
        return parser

    def __init__(
        self,
        load_size,
        crop_size,
        min_scale_crop,
        rotation_degree,
        cutout=False,
        color_distort=False,
        **kwargs,
    ):
        mean = std = (0.5, 0.5, 0.5)
        self.transform = [
            transforms.Compose([
                PadRandomAffine(angle=rotation_degree),
                Resize(load_size, crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            transforms.Compose([
                PadRandomAffine(
                    angle=rotation_degree,
                    scale=(0.8, 1.2),
                ),
                transforms.RandomResizedCrop(
                    size=crop_size,
                    scale=(min_scale_crop, 1.0),
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(.2, .2, .2, .01)],
                    p=0.8 if color_distort else 0.0,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.RandomErasing(.8 if cutout else 0., (0.1, 0.33)),
                transforms.Normalize(mean=mean, std=std),
            ])
        ]

    def __call__(self, x):
        return list(map(lambda t: t(x), self.transform))


class SimpleTransform(transforms.Compose):
    def __init__(self, image_size, interpolation=BICUBIC, **kwargs):
        super().__init__([
            transforms.Resize((image_size,)*2, interpolation),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
