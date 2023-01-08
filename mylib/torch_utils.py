import glob
import os

import torch
import torch.nn as nn


def count_parameters(module):
    assert isinstance(module, nn.Module)
    num_params = 0
    for p in module.parameters():
        if p.requires_grad:
            num_params += p.numel()
    return num_params


@torch.no_grad()
def concat_all_gather(tensor, world_size):
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output


def kaiming_init(module):
    assert isinstance(module, nn.Module)
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in")
        if module.bias is not None and module.bias.requires_grad:
            nn.init.zeros_(module.bias)


def load_checkpoint(run_dir):
    # TODO: add options to choose specific checkpoint.
    filename = os.path.join(run_dir, "*.pt")
    filelist = sorted(glob.glob(filename))
    if not filelist:
        print("Found 0 checkpoints.")
        return None
    checkpoint = filelist[-1]
    print(f"Loading checkpoint '{checkpoint}'")
    checkpoint = torch.load(checkpoint, map_location="cpu")
    return checkpoint


def unnormalize(image, to_uint8=False):
    # assert isinstance(image, torch.Tensor)
    if to_uint8:
        image = image.mul(127.5).add(128).clamp(0, 255)
        return image.to(torch.uint8)
    else:
        return image.mul(0.5).add(0.5).clamp(0, 1)


@torch.no_grad()
def accumulate_weights(net, net_ema, m=0.999):
    net = net.module if hasattr(net, "module") else net
    net_ema = net_ema.module if hasattr(net_ema, "module") else net_ema
    for p, p_ema in zip(net.parameters(), net_ema.parameters()):
        p_ema.mul_(m).add_((1.0 - m) * p.detach())


def warmup_learning_rate(optimizer, lr, train_step, warmup_step):
    # assert isinstance(optimizer, torch.optim.Optimizer)
    if train_step > warmup_step or warmup_step == 0:
        return lr
    ratio = min(1.0, train_step/warmup_step)
    lr_w = ratio * lr
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr_w
    return lr_w


class Queue(nn.Module):
    def __init__(self, queue_size, feature_dim):
        super().__init__()
        self.register_buffer("data", torch.zeros(queue_size, feature_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size
        self.feature_dim = feature_dim
        self.empty = True
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    @torch.no_grad()
    def forward(self, keys):
        # gather from all gpus
        if self.world_size > 1:
            keys = concat_all_gather(keys, self.world_size)
        batch_size = keys.size(0)

        ptr = int(self.ptr)
        assert self.queue_size % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.data[ptr:ptr + batch_size, :] = keys
        self.ptr[0] = (ptr + batch_size) % self.queue_size
        self.empty = False

    def reset(self):
        self.ptr[0] = 0
        self.empty = True

    def extra_repr(self):
        return "queue_size={}, feature_dim={}, ptr={}, empty={}".format(
            self.queue_size, self.feature_dim, int(self.ptr), self.empty
        )
