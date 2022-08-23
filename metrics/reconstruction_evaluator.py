import os
import time

import torch

from mylib.torch_utils import unnormalize
from .base_evaluator import BaseEvaluator


class ReconstructionEvaluator(BaseEvaluator):

    def prepare_evaluation(self):
        print("Preparing reconstruction evaluation ...")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=1, *args, **kwargs):
        print("Evaluating reconstruction ...")
        device = model.device
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True, num_workers=4, drop_last=False
        )

        if hasattr(model, "lpips"):
            lpips_loss_fn = model.lpips
        else:
            import lpips
            lpips_loss_fn = lpips.LPIPS(net="vgg").eval().requires_grad_(False)
            lpips_loss_fn.to(device)
        mse_loss_fn = torch.nn.MSELoss(reduction="mean")

        fname = os.path.join(self.run_dir, "recon/{}.png")
        os.makedirs(os.path.dirname(fname), exist_ok=True)

        lpips_results, mse_results = [], []
        runtimes = []
        for i, x in enumerate(loader):
            x = x.to(device)

            if i == 0:
                dummy_content = model.G_ema(x, command="encode")
                dummy_style = model.D_ema(x, command="encode")
                model.G_ema.clear_noise()
                model.G_ema.fix_noise(dummy_content, dummy_style)

            start_time = time.time()
            x_recon = model.synthesize(x, x)
            runtime = time.time() - start_time

            x_recon = unnormalize(x_recon, to_uint8=True)
            x_recon = x_recon.to(torch.float32) / 255.0

            x = x.add(1.0).mul(0.5)
            lpips_loss = lpips_loss_fn(x, x_recon).mean()
            mse_loss = mse_loss_fn(x, x_recon)

            self.save_image(
                torch.cat((x, x_recon)), fname.format(i),
                unnormalize=False, nrow=x.size(0)
            )

            lpips_results.append(lpips_loss.item())
            mse_results.append(mse_loss.item())
            runtimes.append(runtime)

        mse_mean = sum(mse_results) / len(mse_results)
        lpips_mean = sum(lpips_results) / len(lpips_results)
        runtime_mean = sum(runtimes) / len(runtimes)
        return {
            "LPIPS_recon": lpips_mean,
            "MSE": mse_mean,
            "sec/iter": runtime_mean
        }
