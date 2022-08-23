import os
import shutil

IMG_EXTENSIONS = (
    'jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp',
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


def report_metric(result_dict, run_dir=None):
    line = ",".join([f"{k}={v}" for k, v in result_dict.items()])
    print(line)
    if run_dir is not None and os.path.isdir(run_dir):
        txtfile = os.path.join(run_dir, "metrics.txt")
        with open(txtfile, "at") as f:
            f.writelines([line + "\n"])
