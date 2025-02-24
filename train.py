# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

import torch
from nn import engines


def get_args():
    parser = argparse.ArgumentParser(description="Command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")
    parser.add_argument("--seed", type=int,
                        help="manual seed to be used.", metavar="")

    parser.add_argument("--engine", type=str, required=True,
                        help="engine specification.", metavar="")
    parser.add_argument("--epochs", type=int, default=1,
                        help="how many times to iterate over the dataset (default: 1).", metavar="")
    parser.add_argument("--limit_batches", type=float, default=1.0,
                        help="how much of the dataset to use (default: 1.0).", metavar="")

    return parser.parse_args()


def main(args):
    device = "cuda" if args.use_gpu >= 0 else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 1. Define engine.
    engine = engines.get(args.engine)

    # 2. Invoke engine.
    engine.fit(epochs=args.epochs, limit_batches=args.limit_batches, device=device)


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
