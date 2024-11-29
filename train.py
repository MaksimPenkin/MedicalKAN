# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

from nn import models, callbacks
from data import datasets
from torch.utils.data import DataLoader

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
    parser.add_argument("--val_db", type=str,
                        help="path to the validation dataset configuration file or identifier.", metavar="")

    parser.add_argument("--loss", type=str, default="mse",
                        help="path to the loss configuration file or identifier.", metavar="")
    parser.add_argument("--optimizer", type=str, default="adam",
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
    dataloader = DataLoader(datasets.get(args.db),
                            batch_size=args.batch_size,
                            shuffle=True,
                            pin_memory=True)

    if args.val_db is not None:
        val_dataloader = DataLoader(datasets.get(args.val_db),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    pin_memory=True)
    else:
        val_dataloader = None

    # Model.
    model = models.get(args.nn, checkpoint=args.ckpt)

    # Train!
    train_func(model, dataloader,
               criterion=args.loss,
               optimizer=args.optimizer,
               callbacks=[callbacks.get(load_config(callback)) for callback in args.callbacks],
               epochs=args.epochs,
               val_dataloader=val_dataloader,
               device=device)


if __name__ == "__main__":
    from datetime import datetime
    from utils.torch_utils import torch_device

    print(torch_device())
    print("\nCommand-line arguments:")
    cmd_args = parse_args()
    for k, v in vars(cmd_args).items():
        print(f"\t{k:20}: {v}")

    start_time = datetime.now()
    print(f"\n{start_time}: Script `{os.path.basename(__file__)}` has started.")
    main(cmd_args)
    end_time = datetime.now()
    print(f"\n{end_time}: Script `{os.path.basename(__file__)}` has stopped.\n"
          f"Elapsed time: {end_time - start_time} (hours : minutes : seconds : microseconds).")

