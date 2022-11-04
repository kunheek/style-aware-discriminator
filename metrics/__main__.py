#!/usr/bin/env python
import argparse
import importlib
import os

import torch

import metrics
from mylib import data, misc
from model import StyleAwareDiscriminator
from model.augmentation import SimpleTransform


def parse_args():
    parser = argparse.ArgumentParser()
    for k, v in metrics.__dict__.items():
        if "Evaluator" in k:
            parser = v.add_commandline_args(parser)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--train-dataset")
    parser.add_argument("--eval-dataset")
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint)
    return args


def find_evaluator_using_name(evaluator_name):
    evaluator_filename = "metrics.{}_evaluator".format(evaluator_name)
    evaluatorlib = importlib.import_module(evaluator_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    evaluator = None
    evaluator_name = evaluator_name.replace("_", "")
    for name, cls in evaluatorlib.__dict__.items():
        if name.lower() == evaluator_name + "evaluator":
            evaluator = cls

    if evaluator is None:
        raise ValueError("In %s.py, there should be a class named Evaluator")

    return evaluator


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opts = checkpoint["options"]
    args.image_size = opts.image_size
    args.run_dir = opts.run_dir
    step = checkpoint["step"]
    print(opts)

    # Override options.
    for k, v in vars(args).items():
        if v is not None:
            setattr(opts, k, v)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    misc.set_seed(opts.seed)

    model = StyleAwareDiscriminator(opts)
    model.load(checkpoint)
    del model.optimizer  # NOTE: we don't need optimizer here.
    torch.cuda.empty_cache()

    transform = SimpleTransform(opts.image_size)
    eval_dataset = data.build_dataset(opts.eval_dataset, transform)

    results = {"seed": opts.seed}
    for task in args.tasks:
        evaluator = find_evaluator_using_name(task)(**vars(opts))
        if evaluator.is_available():
            result = evaluator.evaluate(model, eval_dataset, step=step)
            results.update(result)
    misc.report(results, run_dir=opts.run_dir, filename="metrics")


if __name__ == "__main__":
    main()
