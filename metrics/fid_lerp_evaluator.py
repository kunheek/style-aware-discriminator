import os
import shutil

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as VF
from cleanfid import fid

from .fid_evaluator import FIDEvaluator


class FIDLerpEvaluator(FIDEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        return parser

    def prepare_evaluation(self):
        print("Preparing FID_lerp evaluation ...")
        super().prepare_evaluation()

        domains = os.listdir(self.eval_dataset)
        domains.sort()
        print(f"\tNumber of domains: {len(domains)}")

        self.tasks = []
        for ref_domain in domains:
            ref_path = os.path.join(self.eval_dataset, ref_domain)
            ref_dataset = self.get_eval_dataset(ref_path)

            src_domains = [x for x in domains if x != ref_domain]
            for src_domain in src_domains:
                task = f"{src_domain}2{ref_domain}"
                src_path = os.path.join(self.eval_dataset, src_domain)
                src_dataset = self.get_eval_dataset(src_path)

                self.tasks.append({
                    "name": task,
                    "source": src_dataset,
                    "reference": ref_dataset,
                })
        print(f"\tNumber of tasks: {len(self.tasks)}")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=50, *args, **kwargs):
        print("Evaluating FID_lerp ...")
        device = model.device

        n = 0
        path_fake = os.path.join(self.run_dir, "fid_lerp")
        shutil.rmtree(path_fake, ignore_errors=True)
        os.makedirs(path_fake)
        for task in self.tasks:
            src_loader = torch.utils.data.DataLoader(
                task["source"], batch_size=batch_size,
                shuffle=True, num_workers=4, drop_last=True
            )
            ref_loader = torch.utils.data.DataLoader(
                task["reference"], batch_size=batch_size,
                shuffle=True, num_workers=4, drop_last=True
            )
            assert len(src_loader) == len(ref_loader)  # TODO: make it more flexible to data length.

            print(f'Generating images for {task["name"]}...')
            for x_src, x_ref in zip(src_loader, ref_loader):
                x_src = x_src.to(device)
                x_ref = x_ref.to(device)
                s_src = model.D_ema(x_src, command="encode")
                s_ref = model.D_ema(x_ref, command="encode")
                for w in (0.2, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8):
                    s = F.normalize(torch.lerp(s_src, s_ref, w))
                    x_fake = model.synthesize(x_src, s)
                    x_fake = x_fake.mul(127.5).add(128).clamp(0, 255)
                    x_fake = x_fake.to(torch.uint8)

                    # Save generated images to calculate FID later.
                    for i in range(x_src.size(0)):
                        filename = os.path.join(path_fake, f'{n:05d}.png')
                        pil_image = VF.to_pil_image(x_fake[i])
                        pil_image.save(filename)
                        n += 1

            del src_loader
            del ref_loader

        torch.cuda.empty_cache()
        fid_lerp = fid.compute_fid(
            path_fake, dataset_name=self.dataset_name,
            mode="clean", dataset_split="custom",
        )
        return {"FID_lerp": fid_lerp}
