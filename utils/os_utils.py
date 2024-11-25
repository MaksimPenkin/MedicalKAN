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
        raise ValueError(f"Failed to delete: {filename}.")


def reset_dir(directory):
    for filename in os.listdir(directory):
        delete_file(os.path.join(directory, filename))


def make_dir(directory, exist_ok=False, force=False):
    try:
        os.makedirs(directory, exist_ok=exist_ok)
    except FileExistsError:
        if force:
            reset_dir(directory)
        else:
            raise FileExistsError(f"Cannot create directory, that already exists: {directory}.")


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
