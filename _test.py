# """
# @author   Maksim Penkin
# """

import argparse
from argparse import RawTextHelpFormatter

import lightning
from src import trainers, models, data
from src.utils.serialization_utils import load_config


def get_args():
    parser = argparse.ArgumentParser(usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--config", type=str, required=True, help="path to experiment configuration file (*.yaml).", metavar="")

    parser.add_argument("--limit_train_batches", type=float, default=1.0, help="how much of training dataset to use (default: 1.0).", metavar="")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="how much of validation dataset to use (default: 1.0).", metavar="")

    parser.add_argument("--seed", type=int, help="if specified, sets the seed for pseudo-random number generators.", metavar="")

    return parser.parse_args()


import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio
from tqdm import tqdm

def main(args):
    # 1. Setup.
    if args.seed is not None:
        lightning.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)

    # 2.1 Construct model.
    model = models.get(cfg["model"])
    # 2.2 Construct data.
    val_loader = data.get(cfg["data"]["val_dataloaders"][0])  # TODO: fix the issue with multiple loaders.

    # 3. Test.  # TODO: refactor it.
    model = model._model
    model.eval()
    model.to("cuda")
    psnr_list = []
    tv_list = []
    for batch in tqdm(val_loader):
        x, y = batch
        y_pred = model(x.to("cuda"))

        y_pred = np.clip(y_pred.detach().cpu().numpy(), 0., 1.)[0, 0]
        y = np.clip(y.detach().cpu().numpy(), 0., 1.)[0, 0]

        y_pred = (y_pred * 255).astype(np.uint8).astype(np.float32) / 255.
        y = (y * 255).astype(np.uint8).astype(np.float32) / 255.

        psnr = peak_signal_noise_ratio(y, y_pred, data_range=1.0)
        tv = tf.image.total_variation(tf.convert_to_tensor(y_pred[None, ..., None]))

        psnr_list.append(psnr)
        tv_list.append(tv)

    print(np.array(psnr_list).mean())
    print(np.array(tv_list).mean())


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime

    cmd_args = get_args()
    for k, v in vars(cmd_args).items():
        print(f"\t{k:20}: {v}")

    start_time = datetime.now()
    print(f"\n{start_time}: Script `{Path(__file__).name}` has started.")
    main(cmd_args)
    end_time = datetime.now()
    print(
        f"\n{end_time}: Script `{Path(__file__).name}` has stopped.\n"
        f"Elapsed time: {end_time - start_time} (hours : minutes : seconds : microseconds)."
    )
