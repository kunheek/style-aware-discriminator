import os
import shutil

import torch
from cleanfid import fid
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from mylib import misc, torch_utils
from .fid_evaluator import FIDEvaluator


class MeanFIDEvaluator(FIDEvaluator):

    @staticmethod
    def add_commandline_args(parser):
        parser.add_argument("--num-repeat", type=int, default=10, help="mFID")
        parser.add_argument("--eval-mkid", type=misc.str2bool, default=False)
        return parser

    def __init__(self, num_repeat, eval_mkid, **kwargs):
        self.num_repeat = num_repeat
        self.eval_mkid = eval_mkid
        super().__init__(**kwargs)

    def prepare_evaluation(self):
        print("Preparing mFID evaluation ...")
        
        if self.eval_dataset.endswith("/"):
            self.eval_dataset = self.eval_dataset[:-1]
        splits = self.eval_dataset.split("/")
        if splits[-1] in ("eval", "test", "train", "val", "valid"):
            dataset = splits[-2]
        else:
            dataset = splits[-1]
        print(f"\tDataset: {dataset}")

        domains = os.listdir(self.eval_dataset)
        domains.sort()
        print(f"\tNumber of domains: {len(domains)}")
        if len(domains) < 2 or "data.mdb" in domains:
            print("\tmFID evaluation requires more than 2 domains.")
            self._is_available = False
            return self._is_available
        elif len(domains) > 5:
            print("\tmFID evaluation for many domains (>5) not implemented.")
            self._is_available = False
            return self._is_available

        self.tasks = []
        for ref_domain in domains:
            ref_path = os.path.join(self.eval_dataset, ref_domain)
            ref_dataset = self.get_eval_dataset(ref_path)
            dataset_name = f"{dataset}_{ref_domain}{self.image_size}"
            if not fid.test_stats_exists(dataset_name, mode="clean"):
                print(f"Creating custom stats '{dataset_name}'...")
                path_real = os.path.join(self.train_dataset, ref_domain)
                self.make_custom_stats(path_real, dataset_name)
                print("Done!\n")

            src_domains = [x for x in domains if x != ref_domain]
            for src_domain in src_domains:
                task = f"{src_domain}2{ref_domain}"
                src_path = os.path.join(self.eval_dataset, src_domain)
                src_dataset = self.get_eval_dataset(src_path)

                self.tasks.append({
                    "name": task,
                    "source": src_dataset,
                    "reference": ref_dataset,
                    "dataset_name": dataset_name
                })
        print(f"\tNumber of tasks: {len(self.tasks)}")
        print("Done!")
        self._is_available = True
        return self._is_available

    @torch.no_grad()
    def evaluate(self, model, dataset=None, batch_size=50, *args, **kwargs):
        print("Evaluating mFID ...")
        device = model.device
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "num_workers": 4,
            "drop_last": True,
        }

        results_dict = {}
        for task in self.tasks:
            path_fake = os.path.join(self.run_dir, "mfid", task["name"])
            shutil.rmtree(path_fake, ignore_errors=True)
            os.makedirs(path_fake)

            src_loader = DataLoader(task["source"], **loader_kwargs)
            ref_loader = DataLoader(task["reference"], **loader_kwargs)
            # TODO: more flexible to data length.
            assert len(src_loader) == len(ref_loader)

            num_fake = 0
            print(f'Generating images for {task["name"]}...')
            for _ in range(self.num_repeat):
                for x_src, x_ref in zip(src_loader, ref_loader):
                    x_src = x_src.to(device)
                    x_ref = x_ref.to(device)

                    x_fake = model.synthesize(x_src, x_ref)
                    x_fake = torch_utils.unnormalize(x_fake, to_uint8=True)

                    # Save generated images to calculate FID later.
                    for i in range(x_src.size(0)):
                        num_fake += 1
                        filename = os.path.join(path_fake, f"{num_fake:05d}.png")
                        pil_image = to_pil_image(x_fake[i])
                        pil_image.save(filename)

            del src_loader, ref_loader
            results_dict["FID_"+task["name"]] = fid.compute_fid(
                path_fake, dataset_name=task["dataset_name"],
                mode="clean", dataset_split="custom"
            )
            if self.eval_mkid:
                results_dict["KID_"+task["name"]] = fid.compute_kid(
                    path_fake, dataset_name=task["dataset_name"],
                    mode="clean", dataset_split="custom"
                )

        del x_src, x_ref, x_fake
        torch.cuda.empty_cache()

        sum_fid, sum_kid = 0.0, 0.0
        for k, v in results_dict.items():
            if "FID" in k:
                sum_fid += v
            elif "KID" in k:
                sum_kid += v
        results_dict["mFID"] = sum_fid / len(self.tasks)
        if self.eval_mkid:
            results_dict["mKID"] = sum_kid / len(self.tasks)
        return results_dict
