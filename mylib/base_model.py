import torch

from mylib import torch_utils


class BaseModel(torch.nn.Module):
    def __init__(self, options):
        super().__init__()
        assert hasattr(options, "run_dir")
        self.opt = options
        self.device = "cpu"
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.loss = {}
        self._create_networks()
        self._create_criterions()
        self._configure_gpu()
        self._create_optimizers()

    def _create_networks(self):
        raise NotImplementedError

    def _create_criterions(self):
        self.optimizers = {}

    def _configure_gpu(self):
        self.device = torch.device(self.rank)
        self.to(self.device)
        if self.world_size > 1:
            for name, module in self.named_children():
                if torch_utils.count_parameters(module) > 0:
                    module = torch.nn.parallel.DistributedDataParallel(
                        module,
                        device_ids=[self.rank],
                        broadcast_buffers=False,
                        find_unused_parameters=True,
                    )
                    setattr(self, name, module)

    def _create_optimizers(self):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError

    def get_state(self, **kwargs):
        state = {"options": self.opt}
        state.update(**kwargs)

        for name, net in self.named_children():
            net = net.module if hasattr(net, "module") else net
            state[name+"_state_dict"] = net.state_dict()

        for name, optim in self.optimizers.items():
            state[name+"_optimizer"] = optim.state_dict()
        return state

    def load(self, checkpoint=None):
        if checkpoint is None:
            print(f"Checkpoint is None.")
            return

        print(" "*4, f"step={checkpoint['step']}")
        print(" "*4, f"nimg={checkpoint['nimg']}")
        for name, net in self.named_children():
            key = name + "_state_dict"
            if key in checkpoint.keys():
                state_dict = checkpoint[key]
                net = net.module if hasattr(net, "module") else net
                mis, unex = net.load_state_dict(state_dict, False)
                print(" "*4, f"Loaded state dict from {key}")
                num_mis, num_unex = len(mis), len(unex)
                if num_mis > 0:
                    print(" "*8, f"{num_mis} keys are missing.")
                    print(mis)
                if num_unex > 0:
                    print(" "*8, f"{num_unex} unexpected keys in checkpoint.")
                    print(unex)
            else:
                print(f"\tFailed to load {key}")

        for name, opt in self.optimizers.items():
            key = name + "_optimizer"
            if key in checkpoint.keys():
                opt.load_state_dict(checkpoint[key])
                print(" "*4, f"Loaded state dict from {key}")
            else:
                print(" "*4, f"Failed to load {key}")
        print(f"=> Done!")
