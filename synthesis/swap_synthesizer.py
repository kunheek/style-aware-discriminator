import os

import torch

from .base_synthesizer import BaseSynthesizer


class SwapSynthesizer(BaseSynthesizer):

    @staticmethod
    def add_commandline_args(parser):
        return parser

    def prepare_synthesis(self):
        print("Preparing swapping visualization ...")
        if self.folders is None:
            print("'folder' is None.")
            return self._is_available

        self.contents = self.get_dataset(self.folders[0], force_square=False)
        self.styles = self.get_dataset(self.folders[1], force_square=True)

        result_dir = os.path.join(self.run_dir, "swap")
        os.makedirs(result_dir, exist_ok=True)
        self.filename = os.path.join(result_dir, "grid.png")

        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def synthesize(self, model, *args, **kwargs):
        print("Synthesizing images using style code swapping ...")
        device = model.device

        content_images = []
        for content_image in self.contents:
            content_images.append(content_image.unsqueeze(0).to(device))
        
        grid = [torch.ones_like(content_images[0])]
        style_images = []
        for style_image in self.styles:
            style_images.append(style_image)
        style_images = torch.stack(style_images).to(device)
        grid.append(style_images)

        for content_image in content_images:
            inputs = content_image.repeat(style_images.size(0), 1, 1, 1)
            outputs = model.synthesize(inputs, style_images)
            grid.append(content_image)
            grid.append(outputs)

        grid = torch.cat(grid)
        nrow=style_images.size(0) + 1
        self.save_image(grid, self.filename, nrow=nrow)
