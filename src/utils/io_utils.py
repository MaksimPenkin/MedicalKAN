# """
# @author   Maksim Penkin
# """

from pathlib import Path

import numpy as np
import cv2
from scipy.io import loadmat, savemat

from .os_utils import make_dir


def decode_raw(read_path, dtype, shape):
    img = np.fromfile(read_path, dtype=dtype)
    img = np.reshape(img, shape, order="C")
    return img


def decode_mat(read_path, key):
    img = loadmat(read_path)[key]
    img = np.asarray(img)
    return img


def encode_raw(img, save_path):
    np.ravel(img, order="C").tofile(save_path)


def encode_mat(img, save_path, key):
    savemat(save_path, {key: img})
