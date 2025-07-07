# """
# @author   Maksim Penkin
# """

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

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


def main(args):
    # 1. Setup.
    if args.seed is not None:
        lightning.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)

    # 2.1 Construct trainer.
    trainer = trainers.get(cfg["trainer"], limit_train_batches=args.limit_train_batches, limit_val_batches=args.limit_val_batches)
    # 2.2 Construct model.
    model = models.get(cfg["model"])
    # 2.3 Construct data.
    train_loader = data.get(cfg["data"]["train_dataloaders"][0])  # TODO: fix the issue with multiple loaders.
    val_loader = data.get(cfg["data"]["val_dataloaders"][0])  # TODO: fix the issue with multiple loaders.

    # 3. Invoke.
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)


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
