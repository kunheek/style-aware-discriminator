import torch
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

import mylib.data as data


class BaseEvaluator:

    @staticmethod
    def add_commandline_args(parser):
        return parser

    def __init__(
        self, run_dir, train_dataset, eval_dataset,
        image_size, batch_size, **kwargs,
    ):
        self.run_dir = run_dir
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.image_size = image_size
        self.batch_size = batch_size
        self._is_available = False
        self.prepare_evaluation()

    def prepare_evaluation(self):
        raise NotImplementedError

    def evaluate(self, model, dataset, *args, **kwargs):
        raise NotImplementedError

    def is_available(self):
        return self._is_available

    def get_eval_dataset(self, path, return_target=False):
        # We used the PIL-bicubic resampling proposed by Parmar et al.
        # For more detail, please refer to https://arxiv.org/abs/2104.11222.
        interpolation = transforms.InterpolationMode.BICUBIC
        if isinstance(self.image_size, int):
            resize_fn = transforms.Resize((self.image_size,)*2, interpolation)
        else:
            resize_fn = transforms.Compose(
                [
                    transforms.Resize((sz,)*2, interpolation)
                    for sz in set(self.image_size)
                ]
            )
        transform = transforms.Compose([
            resize_fn,
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        return data.ImageFolder(path, transform, return_target)

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
            image = image.mul(127.5).add(128).clamp(0, 255)
        else:
            image = image.mul(255.0).add(0.5).clamp(0, 255)
        image = image.to(torch.uint8)
        pil_image = to_pil_image(image)
        pil_image.save(fname)
