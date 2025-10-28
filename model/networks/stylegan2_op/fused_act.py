import os

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F

# Disable custom CUDA kernels by default for compatibility
# Set USE_CUSTOM_OPS=True to enable custom kernels (not recommended)
USE_CUSTOM_OPS = os.environ.get('USE_CUSTOM_OPS', 'False').lower() == 'true'

if USE_CUSTOM_OPS:
    try:
        from torch.utils.cpp_extension import load
        module_path = os.path.dirname(__file__)
        fused = load(
            "fused",
            sources=[
                os.path.join(module_path, "fused_bias_act.cpp"),
                os.path.join(module_path, "fused_bias_act_kernel.cu"),
            ],
            extra_cflags=["-std=c++17"],
            extra_cuda_cflags=["-std=c++17"],
        )
        print("✓ Using custom CUDA kernels for fused operations")
    except Exception as e:
        print(f"✗ Failed to load custom CUDA kernels: {e}")
        print("→ Falling back to pure PyTorch implementation")
        USE_CUSTOM_OPS = False
        fused = None
else:
    fused = None


class FusedLeakyReLUFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, out, bias, negative_slope, scale):
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        empty = grad_output.new_empty(0)

        grad_input = fused.fused_bias_act(
            grad_output.contiguous(), empty, out, 3, 1, negative_slope, scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        return grad_input, grad_bias

    @staticmethod
    def backward(ctx, gradgrad_input, gradgrad_bias):
        out, = ctx.saved_tensors
        gradgrad_out = fused.fused_bias_act(
            gradgrad_input.contiguous(),
            gradgrad_bias,
            out,
            3,
            1,
            ctx.negative_slope,
            ctx.scale,
        )

        return gradgrad_out, None, None, None, None


class FusedLeakyReLUFunction(Function):
    @staticmethod
    def forward(ctx, input, bias, negative_slope, scale):
        empty = input.new_empty(0)

        ctx.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused.fused_bias_act(input, bias, empty, 3, 0, negative_slope, scale)
        ctx.save_for_backward(out)
        ctx.negative_slope = negative_slope
        ctx.scale = scale

        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors

        grad_input, grad_bias = FusedLeakyReLUFunctionBackward.apply(
            grad_output, out, ctx.bias, ctx.negative_slope, ctx.scale
        )

        if not ctx.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None


def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    # Use pure PyTorch implementation (compatible with all devices and torch versions)
    if input.device.type == "cpu" or not USE_CUSTOM_OPS:
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (
                F.leaky_relu(
                    input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
                )
                * scale
            )
        else:
            return F.leaky_relu(input, negative_slope=negative_slope) * scale
    else:
        # Use custom CUDA kernel (only if explicitly enabled)
        return FusedLeakyReLUFunction.apply(
            input.contiguous(), bias, negative_slope, scale
        )


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
