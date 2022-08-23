import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_2nd_moment(input, dim=1, eps=1e-8):
    return input * input.square().mean(dim, keepdim=True).add(eps).rsqrt()


class LinearLayer(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        bias_init=0.0,
        lr_mul=1.0,
        activation=True,
    ):
        assert 0.0 < lr_mul <= 1.0
        self.lr_mul = lr_mul
        self.bias_init = bias_init
        self.activation = activation
        super().__init__(in_features, out_features, bias)

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight)
        if hasattr(self, "lr_mul") and self.lr_mul < 1.0:
            with torch.no_grad():
                self.weight.div_(self.lr_mul)
        if self.bias is not None:
            nn.init.constant_(self.bias, self.bias_init)

    def forward(self, input):
        if hasattr(self, "lr_mul") and self.lr_mul < 1.0:
            weight = self.weight * self.lr_mul
            if self.bias is not None:
                bias = self.bias * self.lr_mul
        else:
            weight = self.weight
            bias = self.bias
        out = F.linear(input, weight, bias)
        if self.activation:
            out = F.leaky_relu(out, 0.2, True)
        return out


class MappingNetwork(nn.Sequential):
    def __init__(self, in_features, out_features, depth=8):
        layers = []
        for i in range(depth):
            dim_in = in_features if i == 0 else out_features
            layers.append(LinearLayer(dim_in, out_features, lr_mul=0.01))
        super().__init__(*layers)


class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.register_buffer("eps", torch.as_tensor(float(eps)))

    def forward(self, input):
        return normalize_2nd_moment(input, dim=1, eps=self.eps)


class HighPassFilter(nn.Module):
    def __init__(self, w_hpf=1.0):
        super().__init__()
        filter = torch.as_tensor([
            [-1.0, -1.0, -1.0],
            [-1.0, 8.0, -1.0],
            [-1.0, -1.0, -1.0]
        ]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("filter", filter / w_hpf)

    def forward(self, x):
        filter = self.filter.expand(x.size(1), 1, 3, 3)
        return F.conv2d(x, filter, padding=1, groups=x.size(1))
