#!/usr/bin/env python
import argparse
import json
import os
import random

import torch
import torch.distributed as dist
from tqdm import tqdm

import data
import metrics
from mylib import misc, torch_utils
from model import StyleAwareDiscriminator
from model.augmentation import Augmentation, SimpleTransform


def parse_args():
    parser = argparse.ArgumentParser()
    parser = Augmentation.add_commandline_args(parser)
    parser = StyleAwareDiscriminator.add_commandline_args(parser)
    for k, v in metrics.__dict__.items():
        if "Evaluator" in k:
            parser = v.add_commandline_args(parser)

    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--extra-desc", nargs="+", default=[])

    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--total-nimg", type=misc.str2int, default="5M")
    parser.add_argument("--batch-size", type=int, default=32)
    # Datasets.
    parser.add_argument("--train-dataset", default="../datasets/afhq/train")
    parser.add_argument("--eval-dataset", default="../datasets/afhq/val")
    parser.add_argument("--num-workers", type=int, default=4)
    # Logging.
    parser.add_argument("--log-freq", type=int, default=200)
    parser.add_argument("--snapshot-freq", type=int, default=5000)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--save-freq", type=int, default=10000)
    # Evaluations.
    parser.add_argument("--fid-start-after", type=int, default=10000)
    # Misc.
    parser.add_argument("--allow-tf32", type=misc.str2bool, default=False)
    parser.add_argument("--cudnn-bench", type=misc.str2bool, default=True)
    parser.add_argument("--tqdm", type=misc.str2bool, default=True)
    return parser.parse_args()


def options_from_args(args):
    if args.seed is None:
        args.seed = random.randint(0, 999)

    # Create output folder.
    assert os.path.isdir(args.train_dataset)  # TODO: support other formats.
    if "ffhq" in args.train_dataset:
        dataset = "ffhq"
    else:
        splits = "train", "test", "val", "eval", ""
        dataset = args.train_dataset.split("/")
        dataset = [x for x in dataset if x not in splits][-1]
        dataset = dataset.replace("_train", "").replace("_lmdb", "")  # NOTE for LSUNs.
    desc = dataset.replace("_", "") + f"-{args.image_size}x{args.image_size}"
    for d in args.extra_desc:
        desc += f"-{d}"

    prev_run_ids = []
    if os.path.isdir(args.out_dir):
        prev_run_ids = [int(x.split("-")[0]) for x in os.listdir(args.out_dir)]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(args.out_dir, f'{cur_run_id:03d}-{desc}')

    if args.load_size is None or args.load_size < args.image_size:
        args.load_size = args.image_size
    if args.crop_size is None:
        args.crop_size = args.image_size
    return args


def load_options(resume_dir):
    option_file = os.path.join(resume_dir, "options.json")
    assert os.path.isfile(option_file)
    with open(option_file, "r") as f:
        options_dict = json.load(f)
    options = argparse.Namespace(**options_dict)
    options.run_dir = resume_dir
    return options


def training_loop(model, opts, rank, world_size):
    assert opts.batch_size % world_size == 0
    start_step, cur_nimg = 0, 0

    checkpoint = torch_utils.load_checkpoint(opts.run_dir)
    if checkpoint is not None:
        start_step = checkpoint["step"]
        cur_nimg = checkpoint["nimg"]
        model.load(checkpoint)

    datapipe = data.get_dataset(
        opts.train_dataset, Augmentation(**vars(opts)),
        seed=opts.seed, repeat=True,
    )
    datastream = torch.utils.data.DataLoader(
        datapipe, batch_size=opts.batch_size // world_size,
        num_workers=opts.num_workers, pin_memory=True,
    )
    datastream = iter(datastream)

    eval_transform = SimpleTransform(opts.image_size)
    val_dataset = data.get_dataset(opts.eval_dataset, eval_transform)

    iters = opts.total_nimg // opts.batch_size
    pbar = range(start_step+1, iters+1)
    if rank == 0:
        # TODO: migrate to wandb.
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(opts.run_dir)
            tb = True
        except:
            print("Cannot import 'tensorboard'. Skip tensorboard logging.")
            tb = False

        if opts.tqdm:
            pbar = tqdm(pbar)

        # Save training options.
        with open(os.path.join(opts.run_dir, "options.json"), "w") as f:
            json.dump(vars(opts), f, indent=4)

        # Log number of parameters.
        with open(os.path.join(opts.run_dir, "log.txt"), "w") as f:
            lines = ["#Parameters\n"]
            for n, m in model.named_children():
                lines.append(f"{n:>20}: {torch_utils.count_parameters(m)}\n")
            print("".join(lines))
            f.writelines(lines)

        best_fid = float("inf")
        model.prepare_snapshot(val_dataset)

        knn_evaluator = metrics.KNNEvaluator(**vars(opts))
        mfid_evaluator = metrics.MeanFIDEvaluator(**vars(opts))

    for step in pbar:
        xs = next(datastream)
        xs = tuple(map(lambda x: x.to(rank, non_blocking=True), xs))

        model.set_input(step, xs)
        loss = model.training_step()

        if rank != 0:
            continue

        if step%opts.log_freq==0 and tb:
            for k, v in loss.items():
                writer.add_scalar(k, v, step)

        if step % opts.snapshot_freq == 0:
            model.snapshot()

        cur_nimg = step * opts.batch_size
        is_best = False
        if step % opts.eval_freq == 0:
            result_dict = {"step": step, "nimg": f"{cur_nimg//1000}k"}

            if knn_evaluator.is_available():
                knn_dict = knn_evaluator.evaluate(model, step=step)
                result_dict.update(knn_dict)

            if mfid_evaluator.is_available() and step > opts.fid_start_after:
                fid_dict = mfid_evaluator.evaluate(model)
                is_best = fid_dict["mFID"] < best_fid
                best_fid = min(best_fid, fid_dict["mFID"])
                result_dict.update(fid_dict)

            if tb:
                for k, v in result_dict.items():
                    if k == "step":
                        continue
                    writer.add_scalar("Metrics/"+k, v, step)
            misc.report_metric(result_dict, opts.run_dir)

        if step%opts.save_freq==0 or step==iters or is_best:
            fname = f"networks-{cur_nimg//1000:05d}k.pt"
            fname = os.path.join(opts.run_dir, fname)
            torch.save(model.get_state(step=step, nimg=cur_nimg), fname)


def main():
    args = parse_args()
    if args.resume is None:
        options = options_from_args(args)
    else:
        options = load_options(args.resume)
        print(f"resume training '{args.resume}'")

    if "LOCAL_RANK" not in os.environ.keys():
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"=> set cuda device = {rank}")
    torch.cuda.set_device(rank)
    print(f"=> allow tf32 = {options.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = options.allow_tf32
    torch.backends.cudnn.allow_tf32 = options.allow_tf32
    print(f"=> cuDNN benchmark = {options.cudnn_bench}")
    torch.backends.cudnn.benchmark = options.cudnn_bench

    if rank == 0:
        filename = os.path.join(options.run_dir, "code")
        misc.archive_python_files(os.getcwd(), filename)

    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        # Sync options across devices (required for seed and run_dir).
        broadcast_opts = [options]
        dist.broadcast_object_list(broadcast_opts)
        options = broadcast_opts[0]

    print(f"=> random seed = {options.seed}")
    torch_utils.set_seed(options.seed)

    model = StyleAwareDiscriminator(options)
    training_loop(model, options, rank, world_size)


if __name__ == "__main__":
    main()
