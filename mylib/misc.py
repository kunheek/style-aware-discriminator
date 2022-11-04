import importlib.util
import os
import random
import shutil

IMG_EXTENSIONS = (
    "jpg", "jpeg", "png", "ppm", "bmp", "pgm", "tif", "tiff", "webp",
)


def str2bool(input):
    assert isinstance(input, str)
    input = input.strip().lower()
    if input not in ("1", "0", "t", "f", "true", "false"):
        raise ValueError
    return input in ("1", "t", "true")


def str2int(input):
    assert isinstance(input, str)
    input = input.strip().lower().replace(",", "")
    if input.endswith("k"):
        input = float(input.replace("k", "")) * 1e3
    elif input.endswith("m"):
        input = float(input.replace("m", "")) * 1e6
    return int(input)


def archive_python_files(src_dir, output_name, ignore=("runs", "wandb")):
    # TODO: save other languages (e.g., cpp or cu)
    if not isinstance(ignore, (list, tuple)):
        ignore = (ignore,)

    # Copy python files to temporary directory.
    tmp_dir = output_name
    root_dir = os.path.dirname(output_name)
    for path, _, filenames in os.walk(src_dir):
        if not filenames:
            continue

        skip = False
        for ig in ignore:
            if ig in path:
                skip = True
                break
        if skip:
            continue

        pyfiles = list(filter(lambda f: f.endswith(".py"), filenames))
        for file in pyfiles:
            src = os.path.join(src_dir, path, file)
            dst = os.path.join(tmp_dir, path.replace(src_dir, "."))
            os.makedirs(dst, exist_ok=True)
            shutil.copyfile(src, os.path.join(dst, file))

    shutil.make_archive(output_name, format="tar", root_dir=root_dir)
    shutil.rmtree(tmp_dir)


def is_image_file(file):
    return file.split(".")[-1].lower() in IMG_EXTENSIONS


def readable_time(seconds):
    seconds = int(seconds)
    hours = seconds // 3600 % 24
    minutes = seconds // 60 % 60
    seconds = seconds % 60
    return f"{hours:03d}:{minutes:02d}:{seconds:02d}"


def report(result_dict, run_dir=None, filename="report"):
    line = ",".join([f"{k}={v}" for k, v in result_dict.items()])
    print(line)
    if run_dir is not None and os.path.isdir(run_dir):
        txtfile = os.path.join(run_dir, f"{filename}.txt")
        with open(txtfile, "at") as f:
            f.writelines([line + "\n"])


def set_seed(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if importlib.util.find_spec("numpy") is not None:
        import numpy as np
        np.random.seed(seed)
    if importlib.util.find_spec("torch") is not None:
        import torch
        torch.manual_seed(seed)
