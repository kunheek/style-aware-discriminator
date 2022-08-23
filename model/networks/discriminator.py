import torch
import torch.nn as nn
import torch.nn.functional as F

from mylib.torch_utils import kaiming_init


def conv1x1(in_channels, out_channels, **kwargs):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, **kwargs)


def conv3x3(in_channels, out_channels, stride=1, **kwargs):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, **kwargs,
    )


class VectorNorm(nn.Module):
    def forward(self, input):
        assert input.dim() == 2
        return F.normalize(input, p=2, dim=1)


class Projector(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features):
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers = []
        for i, out_feats in enumerate(hidden_features):
            in_feats = in_features if i == 0 else out_feats
            layers.append(nn.Linear(in_feats, out_feats))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Linear(out_feats, out_features))
        layers.append(VectorNorm())
        super().__init__(*layers)


class Discriminator(nn.Sequential):
    def __init__(self, in_features, hidden_features):
        if isinstance(hidden_features, int):
            hidden_features = [hidden_features]
        layers = []
        for i, out_feats in enumerate(hidden_features):
            in_feats = in_features if i == 0 else out_feats
            layers.append(nn.Linear(in_feats, out_feats))
            layers.append(nn.LeakyReLU(0.2, True))
        layers.append(nn.Linear(out_feats, 1))
        super().__init__(*layers)


class DiscriminatorBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsample=False,
        **conv_kwargs,
    ):
        super().__init__()
        self.register_buffer("gain", torch.rsqrt(torch.as_tensor(2.0)))
        activation = nn.LeakyReLU(0.2, inplace=False)
        # residual
        residual = []
        residual.append(conv3x3(in_channels, in_channels, **conv_kwargs))
        residual.append(activation)
        residual.append(conv3x3(in_channels, out_channels, **conv_kwargs))
        residual.append(activation)
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
        x = self.shortcut(x) + self.residual(x)
        return self.gain * x


class StyleDiscriminator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        big_disc=False,
        channel_multiplier=1,
        channel_max=512,
    ):
        super().__init__()
        channels = {
            2**i: min(int(2**(14-i) * channel_multiplier), channel_max)
            for i in range(2, 11)
        }

        activation = nn.LeakyReLU(0.2)
        encoder = [conv1x1(3, channels[image_size]), activation]
        while image_size > 4:
            in_channels = channels[image_size]
            image_size //= 2
            out_channels = channels[image_size]
            encoder.append(
                DiscriminatorBlock(
                    in_channels, out_channels, downsample=True,
                )
            )
        self.feat_dim = out_channels
        encoder.append(conv3x3(out_channels, out_channels))
        encoder.append(activation)

        if big_disc:
            self.encoder = nn.Sequential(*encoder)
            self.projector = Projector(
                out_channels*4*4, out_channels, latent_dim,
            )
            self.discriminator = Discriminator(out_channels*4*4, out_channels)
        else:
            encoder.append(nn.AdaptiveAvgPool2d(output_size=(1,1)))
            self.encoder = nn.Sequential(*encoder)
            self.projector = Projector(out_channels, out_channels, latent_dim)
            self.discriminator = Discriminator(out_channels, out_channels)

        self.apply(kaiming_init)

    def forward(self, *args, command="_forward_impl", **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, input):
        features = self.encoder(input).view(input.size(0), -1)
        embedding = self.projector(features)
        logit = self.discriminator(features)
        return embedding, logit

    def discriminate(self, input):
        return self.discriminator(self.get_features(input))

    def encode(self, input):
        return self.projector(self.get_features(input))

    def get_features(self, input):
        return self.encoder(input).view(input.size(0), -1)
