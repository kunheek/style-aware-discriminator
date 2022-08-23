import torch
import torch.nn as nn
import torch.nn.functional as F

import lpips


def hinge_adversarial_loss(phase, logit_r=None, logit_f=None):
    if phase == "D":
        loss_r = F.relu(1.0 - logit_r).mean()
        loss_f = F.relu(1.0 + logit_f).mean()
        loss = loss_r + loss_f
    else:
        loss = -logit_f.mean()
    return loss


def nonsat_adversarial_loss(phase, logit_r=None, logit_f=None):
    if phase == "D":
        loss_r = F.softplus(-logit_r).mean()
        loss_f = F.softplus(logit_f).mean()
        loss = loss_r + loss_f
    else:
        loss = F.softplus(-logit_f).mean()
    return loss


def get_adversarial_loss(method):
    assert method in ("hinge", "nonsat")
    if method == "hinge":
        return hinge_adversarial_loss
    else:
        return nonsat_adversarial_loss


def compute_grad_gp(d_out, x_in, gamma=1.0, is_patch=False):
    outputs = d_out.sum() if not is_patch else d_out.mean()
    r1_grads = torch.autograd.grad(outputs, inputs=x_in, create_graph=True)[0]
    r1_penalty = r1_grads.square().sum((1,2,3))
    r1_penalty = r1_penalty * (gamma * 0.5)
    return (d_out.mul(0.0) + r1_penalty).mean()  # trick for DDP


def mse_loss(input, target, dim=1):
    loss = (input-target).square().sum(dim=dim)
    return loss.mean()


class SwappedPredictionLoss(nn.Module):
    def __init__(self, temperature, eps=0.05):
        super().__init__()
        self.register_buffer("temperature", torch.as_tensor(temperature))
        self.register_buffer("eps", torch.as_tensor(eps))
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
        else:
            self.world_size = 1

    def forward(self, scores, code_ids, score_ids, bs=999, n_iters=3):
        loss = 0.0
        for s, t in zip(code_ids, score_ids):
            q = self.sinkhorn_knopp(scores[s], n_iters)[:bs]
            logp = F.log_softmax(scores[t][:bs] / self.temperature, dim=1)
            loss -= torch.sum(q * logp, dim=1).mean()
        return loss

    # Code adopted from
    # https://github.com/facebookresearch/swav/blob/master/main_swav.py#L354
    @torch.no_grad()
    def sinkhorn_knopp(self, scores, n_iters):
        Q = torch.exp(scores / self.eps).T  # Q is K-by-B for consistency with notations from our paper
        K = Q.size(0)  # how many prototypes
        B = Q.size(1) * self.world_size # number of samples to assign

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if self.world_size > 1:
            torch.distributed.all_reduce(sum_Q)
        Q /= sum_Q

        for _ in range(n_iters):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)  # u
            if self.world_size > 1:
                torch.distributed.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.T


class ReconstructionLoss(nn.Module):
    def __init__(self, modes, *lpips_args, **lpips_kwargs):
        super().__init__()
        self.modes = list(map(lambda x: x.lower(), modes))
        if "lpips" in self.modes:
            self.lpips = lpips.LPIPS(*lpips_args, **lpips_kwargs)
            self.eval()
            self.requires_grad_(False)

    def forward(self, input, target, *args, **kwargs):
        loss = 0.0
        if "l1" in self.modes:
            loss += F.l1_loss(input, target)
        if "mse" in self.modes:
            loss += F.mse_loss(input, target)
        if "lpips" in self.modes:
            loss += self.lpips(input, target).mean()
        if len(self.modes) > 1:
            loss /= len(self.modes)
        return loss
