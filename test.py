# """
# @author   Maksim Penkin
# """

import os
from tqdm import tqdm
import argparse
from argparse import RawTextHelpFormatter

import torch
import data
from nn import models


def get_args():
    parser = argparse.ArgumentParser(description="Command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")
    parser.add_argument("--seed", type=int,
                        help="manual seed to be used.", metavar="")

    parser.add_argument("--nn", type=str, required=True,
                        help="model specification.", metavar="")
    parser.add_argument("--ckpt", type=str,
                        help="checkpoint to be restored in the model.", metavar="")
    parser.add_argument("--db", type=str, required=True,
                        help="dataloader specification.", metavar="")

    return parser.parse_args()


def main(args):
    device = "cuda" if args.use_gpu >= 0 else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 1. Define model.
    model = models.get(args.engine, checkpoint=args.ckpt)
    model.to(device).eval()

    # 2. Define dataloader.
    db = data.get(args.db)
    steps = len(db)

    # 3. Apply.
    for idx, (x, y) in enumerate(tqdm(db, total=steps)):
        if idx >= steps:
            break
        y_pred = model(x)


if __name__ == "__main__":
    from pathlib import Path
    from datetime import datetime
    from utils.torch_utils import torch_device

    print(torch_device())
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
