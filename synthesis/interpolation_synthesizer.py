import os
import random

import torch
import torch.nn.functional as F

from mylib.misc import str2bool
from .base_synthesizer import BaseSynthesizer


class InterpolationSynthesizer(BaseSynthesizer):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--lerp-mode", choices={"prototype", "reference"})
        parser.add_argument("--lerp-from-src", type=str2bool, default=True)
        return parser

    def __init__(self, lerp_mode, lerp_from_src, **kwargs):
        self.lerp_mode = lerp_mode
        self.lerp_from_src = lerp_from_src
        super().__init__(**kwargs)

    def prepare_synthesis(self):
        print("Preparing interpolation visualization ...")

        if self.lerp_mode is None:
            print("'lerp-mode' is None.")
            return self._is_available

        folder = self.folders[0]
        self.contents = self.get_dataset(folder)
        if self.lerp_mode == "reference":
            self.styles = self.get_dataset(folder, repeat=True)

        self.fig_dir = os.path.join(self.run_dir, "interpolation")
        print(f"\tOutput dir: {self.fig_dir}")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def synthesize(self, model, *args, **kwargs):
        print("Synthesizing images using interpolated style code ...")
        device = model.device
        prototypes = model.prototypes_ema[0].weight.detach()
        nb_proto = prototypes.size(0)

        loader = torch.utils.data.DataLoader(self.contents, batch_size=1)
        if self.lerp_mode == "reference":
            batch_size = 1 if self.lerp_from_src else 2
            ref_loader = torch.utils.data.DataLoader(self.styles, batch_size=batch_size)
            loader = zip(loader, ref_loader)

        os.makedirs(self.fig_dir, exist_ok=True)
        for i, xs in enumerate(loader):
            x_src = xs[0].to(device)
            x_src = x_src.unsqueeze(0) if x_src.dim() == 3 else x_src
            if self.lerp_mode == "reference":
                x_ref = xs[1].to(device)
                x_ref = x_ref.unsqueeze(0) if x_ref.dim() == 3 else x_ref

            # Create the source style code.
            if self.lerp_from_src:
                s_src = model.D_ema(x_src, command="encode")
            elif self.lerp_mode == "prototype":
                id = random.randint(0, nb_proto-1)
                s_src = prototypes[id].unsqueeze(0)
            else:  # lerp_mode: reference
                s_src = model.D_ema(x_ref[0].unsqueeze(0), command="encode")

            # Create the target style code.
            if self.lerp_mode == "prototype":
                id = random.randint(0, nb_proto-1)
                s_ref = prototypes[id].unsqueeze(0)
            else:  # lerp_mode: reference
                s_ref = model.D_ema(x_ref[-1].unsqueeze(0), command="encode")

            s = [s_src]
            for t in (1, 3, 5, 7, 9):
                t *= 0.1
                s.append(F.normalize(torch.lerp(s_src, s_ref, t)))
            s.append(s_ref)
            s = torch.cat(s)
            x_src = x_src.repeat(len(s), 1, 1, 1)
            x_fake = model.synthesize(x_src, s)

            row = torch.cat((x_src[0].unsqueeze(0), x_fake))
            self.save_image(row, os.path.join(self.fig_dir, f"{i:05d}.png"))
