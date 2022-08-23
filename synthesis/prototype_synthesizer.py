import os
import math

import torch

from .base_synthesizer import BaseSynthesizer


class PrototypeSynthesizer(BaseSynthesizer):

    @staticmethod
    def add_commandline_args(parser):
        return parser

    def prepare_synthesis(self):
        print("Preparing prototype synthesis ...")

        if isinstance(self.folders, (list, tuple)):
            folder = self.folders[0]
        if not os.path.isdir(folder) or not os.listdir(folder):
            print("'folder' is None.")
            return self._is_available

        self.contents = self.get_dataset(folder, force_square=False)

        result_dir = os.path.join(self.run_dir, "prototypes")
        os.makedirs(result_dir, exist_ok=True)
        self.fname_content = os.path.join(result_dir, "original_{}.png")
        self.fname = os.path.join(result_dir, "prototypes_{:03d}to{:03d}.png")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def synthesize(self, model, *args, **kwargs):
        print("Synthesizing images using prototypes ...")
        device = model.device

        content_images = []
        for i, content in enumerate(self.contents):
            self.save_image(content, self.fname_content.format(i))
            content_images.append(content.unsqueeze(0).to(device))

        for i, n_proto in enumerate(model.opt.nb_proto):
            for j in range(math.ceil(n_proto/10)):
                begin = j * 10
                end = min(n_proto, (j+1)*10)
                fname = self.fname.format(begin, end-1)
                prototypes = model.prototypes_ema[i].weight[begin:end]

                grid = []
                for content_image in content_images:
                    inputs = content_image.repeat(prototypes.size(0), 1, 1, 1)
                    outputs = model.synthesize(inputs, prototypes)
                    grid.append(content_image)
                    grid.append(outputs)
                grid = torch.cat(grid)
                self.save_image(grid, fname, nrow=prototypes.size(0)+1)
