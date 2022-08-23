#!/usr/bin/env python
import argparse
import importlib
import os

import torch

import synthesis
from model import StyleAwareDiscriminator
from mylib import misc, torch_utils


def parse_args():
    parser = argparse.ArgumentParser()
    for k, v in synthesis.__dict__.items():
        if "Synthesizer" in k:
            parser = v.add_commandline_args(parser)
    parser.add_argument("tasks", nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--folders", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--cudnn-bench", type=misc.str2bool, default=True)
    parser.add_argument("--allow-tf32", type=misc.str2bool, default=False)
    args = parser.parse_args()
    assert os.path.isfile(args.checkpoint)
    return args


def find_synthesizer_using_name(synthesizer_name):
    synthesizer_filename = "synthesis.{}_synthesizer".format(synthesizer_name)
    synthesizerlib = importlib.import_module(synthesizer_filename)

    # In the file, the class called LauncherNameLauncher() will
    # be instantiated. It has to be a subclass of BaseLauncher,
    # and it is case-insensitive.
    synthesizer = None
    synthesizer_name = synthesizer_name.replace("_", "")
    for name, cls in synthesizerlib.__dict__.items():
        if name.lower() == synthesizer_name + "synthesizer":
            synthesizer = cls

    if synthesizer is None:
        raise ValueError("In %s.py, there should be a class named Synthesizer")

    return synthesizer


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    opts = checkpoint["options"]
    args.image_size = opts.image_size
    args.run_dir = opts.run_dir
    print(opts)

    print(f"=> allow tf32 = {opts.allow_tf32}")
    torch.backends.cuda.matmul.allow_tf32 = opts.allow_tf32
    torch.backends.cudnn.allow_tf32 = opts.allow_tf32
    print(f"=> cuDNN benchmark = {opts.cudnn_bench}")
    torch.backends.cudnn.benchmark = opts.cudnn_bench
    print(f"=> random seed = {opts.seed}")
    torch_utils.set_seed(opts.seed)

    model = StyleAwareDiscriminator(opts)
    model.load(checkpoint)
    del model.optimizers
    torch.cuda.empty_cache()

    for task in args.tasks:
        synthesizer = find_synthesizer_using_name(task)(**vars(args))
        if synthesizer.is_available():
            synthesizer.synthesize(model, args.batch_size)


if __name__ == "__main__":
    main()
