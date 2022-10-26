import os
import shutil

import torch
from cleanfid import fid
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from mylib import misc, torch_utils
from .base_evaluator import BaseEvaluator


class FIDEvaluator(BaseEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument(
            "--fid-mode", choices={"latent", "reference"}, default="latent",
        )
        parser.add_argument(
            "--fid-nimg", type=int, default=50_000,
            help="Number of samples to generate for FID/KID."
        )
        parser.add_argument("--eval-kid", type=misc.str2bool, default=False)
        return parser

    def __init__(self, fid_mode, fid_nimg, eval_kid, **kwargs):
        self.fid_mode = fid_mode
        self.fid_nimg = fid_nimg
        self.eval_kid = eval_kid
        super().__init__(**kwargs)

    def prepare_evaluation(self):
        print("Preparing FID evaluation ...")

        if self.eval_dataset.endswith("/"):
            self.eval_dataset = self.eval_dataset[:-1]
        if "afhq" in self.eval_dataset:
            dataset = "afhq"
        elif "celeba_hq" in self.eval_dataset:
            dataset = "celeba_hq"
        elif "church" in self.eval_dataset:
            dataset = "lsun_church"
        elif "ffhq" in self.eval_dataset:
            dataset = "ffhq"
        else:
            raise ValueError(
                f"Unsupported dataset ({self.eval_dataset}). "
                "Supported datasets are: AFHQ, CelebAHQ, LSUN_church, FFHQ"
            )
        print(f"\tDataset: {dataset}")

        self.cleanfid_kwargs = {
            "mode": "clean",
            "dataset_res": self.image_size,
        }
        if dataset == "lsun_church" and self.image_size == 256:
            self.cleanfid_kwargs["dataset_name"] = dataset
            self.cleanfid_kwargs["dataset_split"] = "train"
        elif dataset == "ffhq" and self.image_size in (256, 1024):
            self.cleanfid_kwargs["dataset_name"] = dataset
            self.cleanfid_kwargs["dataset_split"] = "trainval70k"
        else:
            dataset_name = f"{dataset}{self.image_size}"
            self.cleanfid_kwargs["dataset_name"] = dataset_name
            self.cleanfid_kwargs["dataset_split"] = "custom"
            if not fid.test_stats_exists(dataset_name, mode="clean"):
                print(f"Creating custom stats '{dataset_name}'...")
                self.make_custom_stats(self.train_dataset, dataset_name)
                print("Done!\n")

        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=64, *args, **kwargs):
        print("Evaluating FID ...")
        device = model.device
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, drop_last=True,
        )

        path_fake = os.path.join(self.run_dir, "fid")
        shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake)

        num_fake = 0
        print(f'Generating images ..')
        while num_fake < self.fid_nimg:
            for x_src in loader:
                x_src = x_src.to(device)
                if self.fid_mode == "latent":
                    s_ref = model.prototypes_ema.sample(x_src.size(0))
                else:
                    s_ref = model.D_ema(x_src, command="encode")
                    s_ref = s_ref[torch.randperm(x_src.size(0))]

                x_fake = model.synthesize(x_src, s_ref)
                x_fake = torch_utils.unnormalize(x_fake, to_uint8=True)

                for i in range(x_src.size(0)):
                    num_fake += 1
                    filename = os.path.join(path_fake, f'{num_fake:05d}.png')
                    pil_image = to_pil_image(x_fake[i])
                    pil_image.save(filename)

                    if num_fake >= self.fid_nimg:
                        break
                if num_fake >= self.fid_nimg:
                    break

        del loader
        del x_src, s_ref, x_fake
        torch.cuda.empty_cache()

        fid_value = fid.compute_fid(path_fake, **self.cleanfid_kwargs)
        results_dict = {f"FID_{self.fid_mode}": fid_value}
        if self.eval_kid:
            try:
                kid_value = fid.compute_kid(path_fake, **self.cleanfid_kwargs)
                results_dict[f"KID_{self.fid_mode}"] = kid_value
            except:
                print("KID evaluation is not available.")
        return results_dict

    def make_custom_stats(self, path, dataset_name):
        tmpdir = "/tmp/fid_cache"
        shutil.rmtree(tmpdir, ignore_errors=True)
        os.makedirs(tmpdir)

        n = 0
        for root, _, fnames in sorted(os.walk(path, followlinks=True)):
            for fname in fnames:
                if not misc.is_image_file(fname):
                    continue
                image = Image.open(os.path.join(root, fname)).convert("RGB")
                image = image.resize((self.image_size,)*2, resample=Image.BICUBIC)
                image.save(os.path.join(tmpdir, f"{n:05d}.png"))
                n += 1

        fid.make_custom_stats(dataset_name, tmpdir, mode="clean")
        shutil.rmtree(tmpdir)
        return fid.test_stats_exists(dataset_name, mode="clean")
