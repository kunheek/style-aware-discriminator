import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, **kwargs):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, **kwargs,
    )


class ToRGB(nn.Module):
    def __init__(self, in_channels, upsample=False):
        super().__init__()
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = conv1x1(in_channels, 3)

    def forward(self, input, skip=None):
        x = self.conv(input)
        if skip is not None and hasattr(self, "upsample"):
            skip = self.upsample(skip)
            x = x + skip
        return x


class AdaIN2d(nn.Module):
    def __init__(self, in_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(in_features)
        self.fc = nn.Linear(style_dim, in_features*2)
        self.register_buffer("one", torch.ones(1))

    def forward(self, input, style_code, mask=None):
        if mask is None:
            return self._forward(input, style_code)
        else:
            assert input.size(0) * 2 == style_code.size(0)
            if mask.size(2) != input.size(2):
                mask = F.interpolate(mask, size=input.size(2), mode="nearest")
            return self._masked_forward(input, style_code, mask)

    def _forward(self, input, style_code):
        h = self.fc(style_code)
        h = h.view(*h.size(), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (self.one + gamma) * self.norm(input) + beta

    def _masked_forward(self, input, style_codes, mask):
        h = self.fc(style_codes)
        h = h.view(*h.size(), 1, 1)
        h0, h1 = torch.chunk(h, chunks=2)

        x = self.norm(input)

        gamma0, b0 = torch.chunk(h0, chunks=2, dim=1)
        output0 = (self.one + gamma0) * x + b0
        output0 = mask * output0

        gamma1, b1 = torch.chunk(h1, chunks=2, dim=1)
        output1 = (self.one + gamma1) * x + b1
        output1 = (self.one - mask) * output1
        return output0 + output1


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
        residual.append(nn.InstanceNorm2d(in_channels))
        residual.append(activation)
        residual.append(conv3x3(in_channels, in_channels, **conv_kwargs))

        residual.append(nn.InstanceNorm2d(in_channels))
        residual.append(activation)
        residual.append(conv3x3(in_channels, out_channels, **conv_kwargs))
        if downsample:
            residual.append(nn.AvgPool2d(kernel_size=2))

        self.residual = nn.Sequential(*residual)

        # shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(conv1x1(in_channels, out_channels, bias=False))
        if downsample:
            shortcut.append(nn.AvgPool2d(kernel_size=2))
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        out = self.shortcut(x) + self.residual(x)
        return self.gain * out


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
        assert architecture in ("resnet", "skip", "wing")
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.upsample = upsample
        self.architecture = architecture

        self.norm0 = AdaIN2d(in_channels, style_dim)
        self.conv0 = conv3x3(in_channels, out_channels)

        self.norm1 = AdaIN2d(out_channels, style_dim)
        self.conv1 = conv3x3(out_channels, out_channels)

        if architecture == "resnet":
            shortcut = []
            if upsample:
                shortcut.append(nn.Upsample(scale_factor=2, mode="nearest"))
            if in_channels != out_channels:
                shortcut.append(
                    conv1x1(in_channels, out_channels, bias=False)
                )
            self.shortcut = nn.Sequential(*shortcut)
            self.register_buffer("gain", torch.rsqrt(torch.as_tensor(2.0)))
        elif architecture == "skip":
            self.to_rgb = ToRGB(out_channels, upsample=upsample)

    def _residual(self, x, style_code, mask):
        x = self.norm0(x, style_code, mask)
        x = self.activation(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv0(x)

        x = self.norm1(x, style_code, mask)
        x = self.activation(x)
        x = self.conv1(x)
        return x

    def forward(self, x, style_code, skip=None, mask=None):
        out = self._residual(x, style_code, mask)
        if self.architecture == "resnet":
            out = self.gain * (out + self.shortcut(x))
        elif self.architecture == "skip":
            skip = self.to_rgb(out, skip)
        return out, skip
