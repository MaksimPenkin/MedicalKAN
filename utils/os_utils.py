# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

import os, shutil, json, yaml
from importlib import import_module


def delete_file(fp):
    if os.path.isfile(fp) or os.path.islink(fp):
        os.unlink(fp)
    elif os.path.isdir(fp):
        shutil.rmtree(fp)
    else:
        raise ValueError("utils/os_utils.py: def delete_file(...): "
                         f"error: failed to delete: {fp}.")


def reset_folder(folder):
    for filename in os.listdir(folder):
        delete_file(os.path.join(folder, filename))


def create_folder(folder, exist_ok=False, force=False):
    try:
        os.makedirs(folder, exist_ok=exist_ok)
    except FileExistsError:
        if force:
            reset_folder(folder)
        else:
            raise FileExistsError("utils/os_utils.py: def create_folder(...): "
                                  f"error: cannot create folder, that already exists: {folder}. "
                                  "In order to reset the folder, set force=True.")


def copy(source, destination):
    try:
        shutil.copy(source, destination)
    except shutil.SameFileError:
        pass


def dump(obj, fp):
    save_dir = os.path.split(fp)[0]
    if save_dir:  # e.g. os.path.split("name.png") -> '', 'name.png'; os.path.split("./name.png") -> '.', 'name.png'
        create_folder(save_dir, exist_ok=True)
    ext = os.path.splitext(fp)[-1].lower()

    if ext == ".json":
        with open(fp, "w") as f:
            json.dump(obj, f, indent=4)
    elif ext == ".yaml":
        with open(fp, "w") as f:
            yaml.dump(obj, f)
    else:
        raise ValueError("utils/os_utils.py: def dump(...): "
                         f"error: expected `.json` or `.yaml` file, found: {fp}.")


def load(fp):
    ext = os.path.splitext(fp)[-1].lower()

    if ext == ".json":
        with open(fp, "r") as f:
            config = json.load(f)
    elif ext == ".yaml":
        with open(fp, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("utils/os_utils.py: def load(...): "
                         f"error: expected `.json` or `.yaml` file, found: {fp}.")

    return config


def dynamic_import_module(module_name, class_name):
    return getattr(import_module(module_name), class_name)


def confirm():
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Are you sure? [Y/N]: ").lower()
    return answer == "y"
