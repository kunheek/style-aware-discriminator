import torch
import torch.nn as nn
import torch.nn.functional as F

import model.networks.stylegan2_op as ops
from .common import LinearLayer, PixelNorm


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, **kwargs):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, **kwargs,
    )


class Blur(nn.Module):
    def __init__(self, kernel, pad, scale_factor=1):
        super().__init__()
        k = self.make_kernel(kernel)
        if scale_factor > 1:
            kernel = kernel * (scale_factor ** 2)
        self.register_buffer("kernel", k)

        self.pad = pad
        self.up = 1
        self.down = 1

    def forward(self, input):
        return ops.upfirdn2d(input, self.kernel, self.up, self.down, self.pad)

    @staticmethod
    def make_kernel(k):
        k = torch.as_tensor(k, dtype=torch.float32)
        if k.dim() == 1:
            k = k[None, :] * k[:, None]
        k /= k.sum()
        return k


class Upsample(Blur):
    def __init__(self, kernel, factor=2):
        p = len(kernel) - factor
        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2
        super().__init__(kernel, (pad0, pad1), factor)
        self.up = factor


class Downsample(Blur):
    def __init__(self, kernel, factor=2):
        p = len(kernel) - factor
        pad0 = (p + 1) // 2
        pad1 = p // 2
        super().__init__(kernel, (pad0, pad1))
        self.down = factor


class NoiseInjection(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        scale_factor = torch.zeros(1, in_channels, 1, 1)
        self.scale_factor = nn.Parameter(scale_factor)
        self.noise = None
        self.cache_noise = False

    def forward(self, image):
        if self.noise is None:
            batch, _, height, width = image.size()
            noise = torch.randn(batch, 1, height, width, device=image.device)
            if self.cache_noise:
                self.noise = noise
        else:
            noise = self.noise
        return image + self.scale_factor * noise


class EncodeBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=True,
        **conv_kwargs,
    ):
        super().__init__()
        self.register_buffer("gain", torch.rsqrt(torch.as_tensor(2.0)))
        activation = nn.LeakyReLU(0.2, inplace=True)
        # residual
        residual = []
        residual.append(PixelNorm())
        residual.append(activation)
        residual.append(conv3x3(in_channels, in_channels, **conv_kwargs))
        residual.append(PixelNorm())
        residual.append(activation)
        residual.append(conv3x3(in_channels, out_channels, **conv_kwargs))
        if downsample:
            residual.append(Downsample(kernel=[1, 3, 3, 1]))
        self.residual = nn.Sequential(*residual)

        # shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(conv1x1(in_channels, out_channels, bias=False))
        if downsample:
            shortcut.append(Downsample(kernel=[1, 3, 3, 1]))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        x = self.shortcut(x) + self.residual(x)
        return self.gain * x


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        self.upsample = upsample
        self.weight_shape = -1, in_channels, kernel_size, kernel_size
        # register buffers.
        self.register_buffer("eps", torch.as_tensor(1e-8))

        # register parameters.
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_normal_(self.weight, a=0.2)

        self.affine = LinearLayer(style_dim, in_channels, bias_init=1.0)
        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), scale_factor=factor)

    def forward(self, input, style):
        batch, in_channels, height, width = input.size()

        # Step1 - Modulation.
        weight = self.weight.unsqueeze(0)
        style = self.affine(style).view(batch, 1, -1, 1, 1)
        weight = weight * style

        # Step2 - Demodulation.
        if self.demodulate:
            dcoefs = (weight.square().sum((2, 3, 4)) + self.eps).rsqrt()
            weight = weight * dcoefs.view(batch, -1, 1, 1, 1)

        # Step3 - Convolution.
        weight = weight.view(*self.weight_shape)
        input = input.view(1, -1, height, width)
        if self.upsample:
            weight = weight.view(batch, *self.weight_shape)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channels, -1, self.kernel_size, self.kernel_size
            )
            output = F.conv_transpose2d(
                input, weight, stride=2, padding=0, groups=batch
            )
            output = output.view(batch, -1, output.size(2), output.size(3))
            output = self.blur(output)
        else:
            output = F.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            output = output.view(batch, -1, height, width)
        return output


class StyleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **conv_kwargs):
        super().__init__()
        self.conv = ModulatedConv2d(
            in_channels, out_channels, kernel_size, **conv_kwargs,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.noise = NoiseInjection(out_channels)
        self.register_buffer("one", torch.ones(1))

    def forward(self, input, style_code, mask=None):
        if mask is None:
            x = self._conv(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            x = self._masked_conv(input, style_code, mask)
        x = ops.fused_leaky_relu(x, self.bias)
        return x

    def _conv(self, input, style_code):
        x = self.conv(input, style_code)
        x = self.noise(x)
        return x

    def _masked_conv(self, input, style_codes, mask):
        s0, s1 = torch.chunk(style_codes, chunks=2)
        x0 = self.conv(input, s0)
        x0 = self.noise(x0)
        if mask.size(2) != x0.size(2):
            mask = F.interpolate(mask, size=x0.size(2), mode="nearest")
        x0 = mask * x0

        x1 = self.conv(input, s1)
        x1 = self.noise(x1)
        x1 = (self.one - mask) * x1
        return x0 + x1


class StyleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        style_dim,
        upsample=False,
        architecture="skip",
    ):
        super().__init__()
        assert architecture in ("skip",)
        self.architecture = architecture
        self.conv0 = StyleConv(
            in_channels, out_channels, 3,
            style_dim=style_dim, upsample=upsample,
        )
        self.conv1 = StyleConv(
            out_channels, out_channels, 3, style_dim=style_dim, upsample=False
        )

        self.to_rgb = ToRGB(out_channels, style_dim, in_channels!=out_channels)

    def forward(self, x, style_code, skip=None, mask=None):
        x = self.conv0(x, style_code, mask)
        x = self.conv1(x, style_code, mask)
        skip = self.to_rgb(x, style_code, skip, mask=mask)
        return x, skip


class ToRGB(nn.Module):
    def __init__(self, in_channels, style_dim, upsample=True):
        super().__init__()
        if upsample:
            self.upsample = Upsample([1, 3, 3, 1], factor=2)

        self.conv = ModulatedConv2d(in_channels, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.register_buffer("one", torch.ones(1))

    def forward(self, input, style_code, skip=None, mask=None):
        if mask is None:
            out = self.conv(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            if mask.size(2) != input.size(2):
                mask = F.interpolate(mask, size=input.size(2), mode="nearest")
            out = self._masked_conv(input, style_code, mask)
        out = out + self.bias
        if skip is not None and hasattr(self, "upsample"):
            skip = self.upsample(skip)
            out = out + skip
        return out

    def _masked_conv(self, input, style_codes, mask):
        s0, s1 = torch.chunk(style_codes, chunks=2)
        
        x0 = self.conv(input, s0)
        x0 = mask * x0

        x1 = self.conv(input, s1)
        x1 = (self.one - mask) * x1
        return x0 + x1
