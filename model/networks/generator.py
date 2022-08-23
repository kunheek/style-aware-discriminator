import torch
import torch.nn as nn
import torch.nn.functional as F

from mylib.torch_utils import kaiming_init
from .common import HighPassFilter, MappingNetwork, PixelNorm


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)


class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        style_dim,
        f_depth,
        content_resolution,
        architecture="skip",
        mod_type="adain",
        channel_multiplier=1.0,
        channel_max=512,
    ):
        super().__init__()
        assert architecture in ("resnet", "skip", "wing")
        assert mod_type in ("adain", "stylegan2")
        channels = {
            2**i: min(int(2**(14-i) * channel_multiplier), channel_max)
            for i in range(2, 11)
        }

        resolution = image_size
        in_channels = channels[resolution]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if mod_type == "adain":
            from .stargan2_layers import EncodeBlock, StyleBlock
        else:
            from .stylegan2_layers import EncodeBlock, StyleBlock


        # Build encoder.
        self.encoder.append(conv1x1(3, in_channels))
        while resolution > content_resolution:
            resolution //= 2
            out_channels = channels[resolution]
            self.encoder.append(EncodeBlock(in_channels, out_channels))
            in_channels = out_channels

        # Build bottleneck blocks.
        for i in range(3 if architecture == "wing" else 2):
            self.encoder.append(
                EncodeBlock(out_channels, out_channels, downsample=False)
            )
            self.decoder.append(
                StyleBlock(
                    out_channels, out_channels, style_dim,
                    upsample=False, architecture=architecture,
                )
            )
        self.encoder.append(PixelNorm())

        # Build decoder.
        while resolution < image_size:
            resolution *= 2
            out_channels = channels[resolution]
            self.decoder.append(
                StyleBlock(
                    in_channels, out_channels, style_dim,
                    upsample=True, architecture=architecture,
                )
            )
            in_channels = out_channels

        self.architecture = architecture
        if architecture in ("resnet", "wing"):
            self.to_rgb = nn.Sequential(
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_channels, 3, 1)
            )
        self.apply(kaiming_init)
        # MappingNetwork uses own weight initialization.
        self.mapping = MappingNetwork(latent_dim, style_dim, depth=f_depth)
        if architecture == "wing":
            self.hpf = HighPassFilter()
        self.cache = {}

    def forward(self, *args, command="_forward_impl", **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, input, latent, **kwargs):
        x = self.encode(input)
        return self.decode(x, latent, **kwargs)

    def encode(self, x):
        self.cache.clear()
        for module in self.encoder:
            x = module(x)
            resolution = x.size(2)
            if self.architecture == "wing" and resolution in (32, 64, 128):
                self.cache[resolution] = x
        return x

    def decode(self, x, latent, mask=None, heatmap=None):
        skip = None
        if isinstance(latent, (list, tuple)):
            latent = torch.cat(latent)

        style_code = self.mapping(latent)
        for module in self.decoder:
            x, skip = module(x, style_code, skip, mask)
            resolution = x.size(2)
            if heatmap is not None and resolution in (32, 64, 128):
                hm = heatmap[0] if resolution == 32 else heatmap[1]
                hm = F.interpolate(hm, size=resolution, mode="bilinear")
                x = x + self.hpf(hm * self.cache[resolution])
        if self.architecture in ("resnet", "wing"):
            skip = self.to_rgb(x)
        return skip

    def clear_noise(self):
        for block in self.decoder:
            for name, module in block.named_modules():
                if "noise" in name:
                    module.cache_noise = False
                    module.noise = None

    def fix_noise(self, dummy_content, dummy_style):
        for block in self.decoder:
            for name, module in block.named_modules():
                if "noise" in name:
                    module.cache_noise = True
        self.decode(dummy_content, dummy_style)
        for block in self.decoder:
            for name, module in block.named_modules():
                if "noise" in name:
                    module.cache_noise = False

    def style_mixing(self, x, latents, heatmap=None):
        assert isinstance(latents, (list, tuple))
        assert len(latents) == len(self.decoder)
        skip = None

        for module, latent in zip(self.decoder, latents):
            style_code = self.mapping(latent)
            x, skip = module(x, style_code, skip)
            resolution = x.size(2)
            if heatmap is not None and resolution in (32, 64, 128):
                hm = heatmap[0] if resolution == 32 else heatmap[1]
                hm = F.interpolate(hm, size=resolution, mode="bilinear")
                x = x + self.hpf(hm * self.cache[resolution])
        if self.architecture in ("resnet", "wing"):
            skip = self.to_rgb(x)
        return skip
