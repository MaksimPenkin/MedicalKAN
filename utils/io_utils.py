# """
# @author   Maksim Penkin <mapenkin@sberbank.ru>
# """

import os

import numpy as np
import cv2
from scipy.io import loadmat

from utils.os_utils import create_folder


def check3dimage(img):
    img = np.asarray(img)

    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim == 3:
        if img.shape[-1] not in [1, 3]:
            raise ValueError("utils/io_utils.py: def check3dimage(...): "
                             f"error: expected 1-channel or 3-channel 3 dimensional image, found shape: {img.shape}.")
    else:
        raise ValueError("utils/io_utils.py: def check3dimage(...): "
                         f"error: expected 2 or 3 dimensional image, found shape: {img.shape}.")

    return img


def decode_png(read_path):
    img = cv2.imread(read_path, -1)
    img = check3dimage(img)
    if (img.ndim == 3) and (img.shape[-1] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def decode_raw(read_path, dtype, shape):
    img = np.fromfile(read_path, dtype=dtype)
    img = np.reshape(img, shape, order="C")
    return img


def decode_mat(read_path, key=None):
    img = loadmat(read_path)
    if key is not None:
        img = img[key]
    return img


def encode_png(img, save_path):
    img = check3dimage(img)
    if (img.ndim == 3) and (img.shape[-1] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img)


def encode_raw(img, save_path):
    np.ravel(img, order="C").tofile(save_path)


def read_img(read_path, dtype=None, shape=None, normalize=None):
    # Setup.
    if not os.path.exists(read_path):
        raise ValueError("utils/io_utils.py: def read_img(...): "
                         f"error: `read_path` does not exist: {read_path}.")
    ext = os.path.splitext(read_path)[-1].lower()

    # Read.
    if ext in [".png", ".jpeg", ".jpg"]:
        img = decode_png(read_path)
    elif ext in [".raw", ".bin"]:
        img = decode_raw(read_path, dtype, shape)
    else:
        raise ValueError("utils/io_utils.py: def read_img(...): "
                         f"error: unrecognized filename extension found: {read_path}. "
                         "Only `.png`, `.jpeg`, `.jpg`, `.raw` or `.bin` are supported.")

    # Process.
    if dtype is not None:
        assert img.dtype == np.dtype(dtype), f"Error: `dtype` mismatch: `{img.dtype}` != `{dtype}`."
    if shape is not None:
        if all(shape):  # Do not assert, if shape is only partly known, e.g. [None, None, 3].
            assert img.shape == tuple(shape), f"Error: `shape` mismatch: `{img.shape}` != `{shape}`."
    if normalize is not None:
        img = img / float(normalize)

    return img


def save_img(img, save_path):
    # Setup.
    save_dir = os.path.split(save_path)[0]
    if save_dir:  # e.g. os.path.split("name.png") -> '', 'name.png'; os.path.split("./name.png") -> '.', 'name.png'
        create_folder(save_dir, exist_ok=True)
    ext = os.path.splitext(save_path)[-1].lower()

    # Save.
    if ext in [".png", ".jpeg", ".jpg"]:
        encode_png(img, save_path)
    elif ext in [".raw", ".bin"]:
        encode_raw(img, save_path)
    else:
        raise ValueError("utils/io_utils.py: def save_img(...): "
                         f"error: unrecognized filename extension found: {save_path}. "
                         "Only `.png`, `.jpeg`, `.jpg`, `.raw` or `.bin` are supported.")
