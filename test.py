# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

from tqdm import tqdm
import numpy as np
import torch

from nn import models
from data.ixi import ixi

from utils.io_utils import maxmin_norm, save_img
from utils.os_utils import create_folder


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")

    parser.add_argument("--nn", type=str,
                        required=True,
                        help="path to the model configuration file or identifier.", metavar="")
    parser.add_argument("--ckpt", type=str,
                        help="path to the checkpoint to be restored in the model.", metavar="")

    parser.add_argument("--db", type=str,
                        required=True,
                        help="path to the dataset configuration file or identifier.", metavar="")

    parser.add_argument("--output", type=str, default="./_output",
                        help="path to the output.", metavar="")

    return parser.parse_args()


def save_result(x, y, output, imname, save_path):
    imname = os.path.splitext(imname)[0]
    create_folder(save_path, exist_ok=True)
    create_folder(os.path.join(save_path, "x_y"), exist_ok=True)
    create_folder(os.path.join(save_path, "x_output"), exist_ok=True)

    z1 = maxmin_norm(np.abs(x - y)) * 255.
    z2 = maxmin_norm(np.abs(x - output)) * 255.

    save_img(output, os.path.join(save_path, f"{imname}.mat"), key="image")
    save_img(z1.astype(np.uint8),
             os.path.join(save_path, "x_y", f"{imname}.png"))
    save_img(z2.astype(np.uint8),
             os.path.join(save_path, "x_output", f"{imname}.png"))


def main(args):
    device = "cuda" if args.use_gpu >= 0 else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    # Dataset.
    dataloader = ixi(args.db,
                     key=("sketch", "image"),
                     batch_size=1,
                     shuffle=False,
                     pin_memory=True)

    # Model.
    model = models.get(args.nn, checkpoint=args.ckpt)

    # Eval!
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y, imname in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            save_result(x.detach().cpu().numpy()[0].transpose((1, 2, 0)),
                        y.detach().cpu().numpy()[0].transpose((1, 2, 0)),
                        output.detach().cpu().numpy()[0].transpose((1, 2, 0)),
                        imname[0],
                        args.output)


if __name__ == "__main__":
    main(parse_args())
