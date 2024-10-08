# """
# @author   Maksim Penkin
# """

import os, shutil


def delete_file(filename):
    if os.path.isfile(filename) or os.path.islink(filename):
        os.unlink(filename)
    elif os.path.isdir(filename):
        shutil.rmtree(filename)
    else:
        raise ValueError("utils/os_utils.py: def delete_file(...): "
                         f"error: failed to delete: {filename}.")


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


def confirm():
    answer = ""
    while answer not in ["y", "n"]:
        answer = input("Are you sure? [Y/N]: ").lower()
    return answer == "y"
