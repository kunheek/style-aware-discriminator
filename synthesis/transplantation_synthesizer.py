import os

import torch
import torch.nn.functional as F
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

from mylib import torch_utils
from .base_synthesizer import BaseSynthesizer


class TransplantationSynthesizer(BaseSynthesizer):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--xywh", type=int, nargs="+", default=[])
        return parser

    def __init__(self, xywh, **kwargs):
        self.xywh = xywh
        super().__init__(**kwargs)

    def prepare_synthesis(self):
        print("Preparing content transplantation ...")
        if self.folders is None:
            print("'folder' is None.")
            return self._is_available
        elif len(self.xywh) != 4:
            print("'xywh' is None.")
            return self._is_available

        result_dir = os.path.join(self.run_dir, "transplantation")
        os.makedirs(result_dir, exist_ok=True)
        self.fname = os.path.join(result_dir, "{}.png")
        self.images = self.get_dataset(self.folders[0], force_square=True)
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def synthesize(self, model, *args, **kwargs):
        print("Synthesizing images using content transplantation ...")
        device = model.device

        # Decide the cnt_res.
        x_tmp = self.images[0].unsqueeze(0).to(device)
        cnt_code = model.G_ema(x_tmp, command="encode")
        cnt_res = cnt_code.size(-1)

        if not self.xywh:
            mask = model.generate_mask(1, cnt_res, halfmask=False)
        else:
            x, y, w, h = self.xywh
            mask_size = (1, 1, cnt_res, cnt_res)
            mask = torch.zeros(mask_size, device=device)
            mask[:, :, y:y+h, x:x+w] = 1.0

        scale = self.image_size // cnt_res
        images = []
        for i, image in enumerate(self.images[:2]):
            images.append(image.unsqueeze(0).to(device))

            # draw box
            pil_image = to_pil_image(torch_utils.unnormalize(image))
            x0, y0 = x * scale, y * scale
            x1, y1 = (x + w) * scale, (y + h) * scale
            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(((x0, y0), (x1, y1)), outline=(255,0,0))
            pil_image.save(self.fname.format(i))

        content1 = model.G_ema(images[0], command="encode")
        content2 = model.G_ema(images[1], command="encode")
        # style1 = model.D_ema(images[0], command="encode")
        style2 = model.D_ema(images[1], command="encode")

        content = mask * content1 + (1.0 - mask) * content2
        output = model.G_ema(content, style2, command="decode")
        # styles = (style1, style2)
        # output = model.G_ema(content, styles, command="decode", mask=mask)

        mask = F.interpolate(mask, size=self.image_size)
        self.save_image(mask, self.fname.format("mask"))
        self.save_image(output, self.fname.format("result"))
