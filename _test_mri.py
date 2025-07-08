# """
# @author   Maksim Penkin
# """

import argparse
from argparse import RawTextHelpFormatter

import numpy as np
import tensorflow as tf
from src import models, data
from src.utils.serialization_utils import load_config

from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio


def get_args():
    parser = argparse.ArgumentParser(usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--config", type=str, required=True, help="path to experiment configuration file (*.yaml).", metavar="")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="how much of validation dataset to use (default: 1.0).", metavar="")

    return parser.parse_args()


def main(args):
    # 1. Setup.
    cfg = load_config(args.config)

    # 2.1 Construct model.
    model = models.get(cfg["model"])
    # 2.2 Construct data.
    val_loader = data.get(cfg["data"]["val_dataloaders"][0])  # TODO: fix the issue with multiple loaders.

    # 3. Test.  # TODO: refactor it.
    model = model._model
    model.eval()
    model.to("cuda")

    metrics_list = {'psnr': [], 'tv': []}

    for batch in tqdm(val_loader):
        x, y = batch["image"], batch["mask"]
        y_pred = model(x.to("cuda"))

        y_pred = np.clip(y_pred.detach().cpu().numpy(), 0., 1.)[0, 0]
        y = np.clip(y.detach().cpu().numpy(), 0., 1.)[0, 0]

        y_pred = (y_pred * 255).astype(np.uint8).astype(np.float32) / 255.
        y = (y * 255).astype(np.uint8).astype(np.float32) / 255.

        psnr = peak_signal_noise_ratio(y, y_pred, data_range=1.0)
        tv = tf.image.total_variation(tf.convert_to_tensor(y_pred[None, ..., None]))

        metrics_list['psnr'].append(psnr)
        metrics_list['tv'].append(tv)

    for k, v in metrics_list.items():
        print(f"Avg. {k}: {np.mean(v):.4f}")


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
