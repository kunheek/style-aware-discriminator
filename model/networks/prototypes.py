import torch
import torch.nn as nn
import torch.nn.functional as F

from mylib.torch_utils import Queue


class Prototypes(nn.Linear):
    def __init__(
        self,
        in_features,
        num_prototype,
        num_queue=0,
        queue_size=0,
    ):
        super().__init__(in_features, num_prototype, bias=False)
        self.num_prototypes = num_prototype
        self.normalize()

        self.queues = nn.ModuleList()
        if queue_size > 0:
            for _ in range(num_queue):
                self.queues.append(Queue(queue_size, in_features))

    def forward(self, *args, command="_forward_impl", **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, input, queue_ids=[None]):
        if self.queues:
            if not isinstance(input, (list, tuple)):
                input = (input,)
            return self._queue_forward(input, queue_ids)
        elif isinstance(input, (list, tuple)):
            scores = super().forward(torch.cat(input))
            return torch.chunk(scores, chunks=len(input))
        return super().forward(input)

    def _queue_forward(self, input, queue_ids=[None]):
        assert len(input) == len(queue_ids)
        scores = []
        for feat, i in zip(input, queue_ids):
            score = super().forward(feat)
            if i is None or self.queues[i].empty:
                scores.append(score)
            else:
                with torch.no_grad():
                    score_q = super().forward(self.queues[i].data)
                scores.append(torch.cat((score, score_q)))
            if i is not None:
                self.queues[i](feat)
        return scores

    @torch.no_grad()
    def normalize(self):
        w = self.weight.data.clone()
        w = F.normalize(w, p=2, dim=1)
        self.weight.copy_(w)

    @torch.no_grad()
    def interpolate(self, indices, weights=(0.1, 0.3, 0.5, 0.7, 0.9)):
        assert len(indices) == 2
        z1, z2 = self.weight.data[indices].clone().detach()

        zs = []
        for w in weights:
            z_lerp = torch.lerp(z1, z2, w)
            zs.append(z_lerp)
        zs = torch.stack(zs)
        return F.normalize(zs, p=2, dim=1)

    @torch.no_grad()
    def sample(self, batch_size, proto_ids=None, target_ids=None, mode=""):
        assert mode in ("", "interpolation", "perturbation")
        if mode == "":
            mode = "interpolation" if torch.rand(1) < 0.5 else "perturbation"

        if proto_ids is None:
            proto_ids = torch.randint(0, self.num_prototypes, (batch_size,))
        prototypes = self.weight.data[proto_ids].clone().detach()

        # sample targets
        if mode == "interpolation":
            if target_ids is None:
                target_ids = torch.randint(0, self.num_prototypes, (batch_size,))
            targets = self.weight.data[target_ids].clone().detach()
            weights = torch.rand((batch_size, 1), device=prototypes.device)
            samples = torch.lerp(prototypes, targets, weights)
        else:  # perturbation
            eps = 0.01 * torch.randn_like(prototypes)
            samples =  prototypes + eps
        return F.normalize(samples, p=2, dim=1)


class MultiPrototypes(nn.ModuleList):
    def __init__(
        self,
        in_features,
        num_prototypes,
        num_queue=0,
        queue_sizes=0,
    ):
        assert type(num_prototypes) == type(queue_sizes)
        if isinstance(num_prototypes, int):
            num_prototypes = [num_prototypes]
            queue_sizes = [queue_sizes]
        assert len(num_prototypes) == len(queue_sizes)
        
        prototypes = []
        for n_proto, q_size in zip(num_prototypes, queue_sizes):
            prototypes.append(
                Prototypes(in_features, n_proto, num_queue, q_size)
            )
        super().__init__(prototypes)

    def forward(self, *args, command="_forward_impl", **kwargs):
        return getattr(self, command)(*args, **kwargs)

    def _forward_impl(self, *args, **kwargs):
        outputs = []
        for module in self:
            outputs.append(module(*args, **kwargs))
        return outputs

    def normalize(self):
        for module in self:
            module.normalize()

    def interpolate(self, *args, id=0, **kwargs):
        return self[id].interpolate(*args, **kwargs)

    def sample(self, *args, id=0, **kwargs):
        return self[id].sample(*args, **kwargs)
