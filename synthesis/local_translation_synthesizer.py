import os

import torch
import torch.nn.functional as F

from mylib.misc import str2bool
from .base_synthesizer import BaseSynthesizer


class LocalTranslationSynthesizer(BaseSynthesizer):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--half-mask", type=str2bool, default=True)
        return parser

    def __init__(self, half_mask, **kwargs):
        self.half_mask = half_mask
        super().__init__(**kwargs)

    def prepare_synthesis(self):
        print("Preparing local image translation ...")

        self.images = self.get_dataset(self.folders[0], force_square=True)

        out_dir = os.path.join(self.run_dir, "local_translation/")
        os.makedirs(out_dir, exist_ok=True)
        self.fname_cnt = os.path.join(out_dir, "{:03d}_cnt.png")
        self.fname_ref1 = os.path.join(out_dir, "{:03d}_ref1.png")
        self.fname_ref2 = os.path.join(out_dir, "{:03d}_ref2.png")
        self.fname_mask = os.path.join(out_dir, "{:03d}_mask.png")
        self.fname_fake = os.path.join(out_dir, "{:03d}_fake.png")
        print(f"\tOutput dir: {out_dir}")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def synthesize(self, model, batch_size=8, *args, **kwargs):
        assert batch_size % 2 == 0
        print("Synthesizing images using local translation ...")
        device = model.device
        loader = torch.utils.data.DataLoader(
            self.images, batch_size=batch_size,
            shuffle=True, num_workers=4, drop_last=False,
        )

        cnt_res = None
        n_img = 0
        for x in loader:
            x_src = x.to(device)
            if cnt_res is None:
                cnt_code = model.G_ema(x_src, command="encode")
                cnt_res = cnt_code.size(-1)

            batch_size = x_src.size(0)
            x_ref1 = x_src[torch.randperm(x_src.size(0))]
            x_ref2 = x_src[torch.randperm(x_src.size(0))]
            if hasattr(model, "fan"):
                heatmap = model.fan.get_heatmap(x)
            else:
                heatmap = None

            s2_index = list(range(batch_size))
            s2_index.reverse()
            # Spatial style mixing.
            s1 = model.D_ema(x_ref1, command="encode")
            s2 = model.D_ema(x_ref2, command="encode")
            mask = model.generate_mask(batch_size, cnt_res, halfmask=self.half_mask)

            x_fake = model.G_ema(x_src, [s1, s2], mask=mask, heatmap=heatmap)

            mask = F.interpolate(mask, size=x.size(2))
            for i in range(batch_size):
                self.save_image(x_src[i], self.fname_cnt.format(n_img))
                self.save_image(x_ref1[i], self.fname_ref1.format(n_img))
                self.save_image(x_ref2[i], self.fname_ref2.format(n_img))
                self.save_image(mask[i], self.fname_mask.format(n_img), False)
                self.save_image(x_fake[i], self.fname_fake.format(n_img))
                n_img += 1
