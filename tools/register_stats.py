#!/usr/bin/env python
import argparse
import glob
import os
import shutil

import cleanfid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir")
    args = parser.parse_args()
    assert os.path.isdir(args.stats_dir)
    return args


def main():
    args = parse_args()
    stats = glob.glob(os.path.join(args.stats_dir, "*.npz"))
    if not stats:
        raise RuntimeError("0 stats found in stats-dir.")

    stats_dir = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    os.makedirs(stats_dir, exist_ok=True)
    for stat in stats:
        shutil.copy(stat, stats_dir)


if __name__ == "__main__":
    main()
