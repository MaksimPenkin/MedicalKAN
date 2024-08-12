# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

from nn import models, callbacks
from data.ixi import ixi

from utils.torch_utils import train_func
from utils.serialization_utils import load_config


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

    parser.add_argument("--loss", type=str,
                        help="path to the loss configuration file or identifier.", metavar="")
    parser.add_argument("--optimizer", type=str,
                        help="path to the optimizer configuration file or identifier.", metavar="")
    parser.add_argument("--callbacks", nargs="+", type=str, default=[],
                        help="list of the callbacks configuration files to be applied.", metavar="")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size.", metavar="")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs.", metavar="")

    return parser.parse_args()


def main(args):
    device = "cuda" if args.use_gpu >= 0 else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    # Dataset.
    dataloader = ixi(args.db,
                     key=("sketch", "image"),
                     batch_size=args.batch_size,
                     shuffle=True,
                     pin_memory=True)

    # Model.
    model = models.get(args.nn, checkpoint=args.ckpt)

    # Train!
    train_func(model, dataloader,
               criterion=args.loss,
               optimizer=args.optimizer,
               callbacks=[callbacks.get(load_config(callback)) for callback in args.callbacks],
               epochs=args.epochs,
               val_dataloader=None,
               device=device)


if __name__ == "__main__":
    from utils.torch_utils import torch_device

    print(torch_device())
    main(parse_args())
