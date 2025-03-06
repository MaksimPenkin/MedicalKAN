# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

import lightning
from src import trainers, models, data
from src.utils.serialization_utils import load_config


def get_args():
    parser = argparse.ArgumentParser(description="Command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")
    parser.add_argument("--seed", type=int,
                        help="manual seed to be used.", metavar="")
    parser.add_argument("--config", type=str, required=True,
                        help="path to an experiment configuration file in yaml (or json) format.", metavar="")

    return parser.parse_args()


def main(args):
    accelerator = "gpu" if args.use_gpu >= 0 else "cpu"
    if accelerator == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    if args.seed is not None:
        lightning.seed_everything(args.seed, workers=True)

    cfg = load_config(args.config)

    # 1. Construct.
    trainer = trainers.get(cfg["trainer"])
    model = models.get(cfg["model"])
    train_dataloaders = [data.get(cfg_db) for cfg_db in cfg["data"]["train_dataloaders"]]
    val_dataloaders = [data.get(cfg_db) for cfg_db in cfg["data"]["val_dataloaders"]]

    # 2. Invoke.
    trainer.fit(model=model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders)


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime

    print("\nCommand-line arguments:")
    cmd_args = get_args()
    for k, v in vars(cmd_args).items():
        print(f"\t{k:20}: {v}")

    start_time = datetime.now()
    print(f"\n{start_time}: Script `{Path(__file__).name}` has started.")
    main(cmd_args)
    end_time = datetime.now()
    print(f"\n{end_time}: Script `{Path(__file__).name}` has stopped.\n"
          f"Elapsed time: {end_time - start_time} (hours : minutes : seconds : microseconds).")
