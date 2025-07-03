# """
# @author   Maksim Penkin
# """

import argparse
from argparse import RawTextHelpFormatter

import numpy as np
from torch.nn import functional as F
from src import models, data
from src.utils.serialization_utils import load_config

from tqdm import tqdm
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision


def get_args():
    parser = argparse.ArgumentParser(usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--config", type=str, required=True, help="path to experiment configuration file (*.yaml).", metavar="")
    parser.add_argument("--limit_val_batches", type=float, default=1.0, help="how much of validation dataset to use (default: 1.0).", metavar="")

    return parser.parse_args()


def medseg_indicators(target, output):

    output_ = output > 0.5
    target_ = target > 0.5

    iou_ = jc(output_, target_)
    dice_ = dc(output_, target_)
    try:
        hd_ = hd(output_, target_)
        hd95_ = hd95(output_, target_)
    except:
        hd_ = 0
        hd95_ = 0
    recall_ = recall(output_, target_)
    specificity_ = specificity(output_, target_)
    precision_ = precision(output_, target_)
    try:
        f1_score = 2 * recall_ * precision_ / (precision_ + recall_)
    except:
        f1_score = 0
    return {'IoU': iou_, 'dice': dice_, 'hd': hd_, 'hd95': hd95_,
            'recall': recall_, "specificity": specificity_, "precision": precision_, 'f1_score': f1_score}


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

    metrics_list = {'IoU': [], 'dice': [], 'hd': [], 'hd95': [],
                    'recall': [], "specificity": [], "precision": [], 'f1_score': []}

    for batch in tqdm(val_loader):
        x, y = batch
        y_pred = F.sigmoid(model(x.to("cuda")))

        y = y.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        m = medseg_indicators(y, y_pred)
        for k in m:
            metrics_list[k].append(m[k])

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
