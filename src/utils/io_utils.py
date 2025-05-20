# """
# @author   Maksim Penkin
# """

import numpy as np
from scipy.io import loadmat, savemat


def decode_raw(read_path, dtype, shape, **kwargs):
    x = np.fromfile(read_path, dtype=dtype, **kwargs)
    x = np.reshape(x, shape, order="C")
    return x


def decode_mat(read_path, key, **kwargs):
    x = loadmat(read_path, **kwargs)[key]
    x = np.asarray(x)
    return x


def decode_npy(read_path, **kwargs):
    x = np.load(read_path, **kwargs)
    return x


def encode_raw(x, save_path, **kwargs):
    np.ravel(x, order="C").tofile(save_path, **kwargs)


def encode_mat(x, save_path, key, **kwargs):
    savemat(save_path, {key: x}, **kwargs)


def encode_npy(x, save_path, **kwargs):
    np.save(save_path, x, **kwargs)
