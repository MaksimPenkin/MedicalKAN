# """
# @author   Maksim Penkin
# """

from pathlib import Path

import numpy as np
import cv2
from scipy.io import loadmat, savemat

from utils.os_utils import make_dir


def _as_hwc(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    elif img.ndim != 3:
        raise ValueError(f"Expected 2 or 3 dimensional image, found ndim: {img.ndim}.")
    return img


def decode_png(read_path):
    img = cv2.imread(read_path, -1)
    img = _as_hwc(img)
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img


def decode_raw(read_path, dtype, shape):
    img = np.fromfile(read_path, dtype=dtype)
    img = np.reshape(img, shape, order="C")
    return img


def decode_mat(read_path, key):
    img = loadmat(read_path)[key]
    img = np.asarray(img)
    return img


def encode_png(img, save_path):
    img = _as_hwc(img)
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(save_path, img)


def encode_raw(img, save_path):
    np.ravel(img, order="C").tofile(save_path)


def encode_mat(img, save_path, key):
    savemat(save_path, {key: img})


def read_img(read_path, dtype=None, shape=None, key=None, normalize=None):
    # Setup.
    read_path = Path(read_path)
    if not read_path.exists():
        raise ValueError(f"Source path does not exist: {read_path}.")
    ext = read_path.suffix.lower()

    # Read.
    if ext in (".png", ".jpeg", ".jpg"):
        img = decode_png(read_path)
    elif ext in (".raw", ".bin"):
        img = decode_raw(read_path, dtype, shape)
    elif ext in (".mat", ):
        img = decode_mat(read_path, key)
    else:
        raise ValueError(f"Unrecognized filename extension found: {read_path}. "
                         "Only `.png`, `.jpeg`, `.jpg`, `.raw`, `.bin` or `.mat` are supported.")

    # Process.
    if dtype is not None:
        assert img.dtype == np.dtype(dtype), f"`dtype` mismatch: `{img.dtype}` != `{dtype}`."
    if shape is not None:
        if all(shape):  # Do not assert, if shape is only partly known, e.g. [None, None, 3].
            assert img.shape == tuple(shape), f"`shape` mismatch: `{img.shape}` != `{shape}`."
    if normalize is not None:
        img = img / float(normalize)

    return img


def save_img(img, save_path, key=None):
    # Setup.
    save_path = Path(save_path)
    make_dir(save_path.parent, exist_ok=True)
    ext = save_path.suffix.lower()

    # Save.
    if ext in (".png", ".jpeg", ".jpg"):
        encode_png(img, save_path)
    elif ext in (".raw", ".bin"):
        encode_raw(img, save_path)
    elif ext in (".mat", ):
        encode_mat(img, save_path, key)
    else:
        raise ValueError(f"Unrecognized filename extension found: {save_path}. "
                         "Only `.png`, `.jpeg`, `.jpg`, `.raw`, `.bin` or `.mat` are supported.")
