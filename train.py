"""
 @author   Maksim Penkin

"""

import os, sys
import argparse
from argparse import RawTextHelpFormatter
sys.path.append('../')

from datetime import datetime
from tqdm import tqdm

import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets.ixi import MatDataset

from torch.utils.tensorboard import SummaryWriter

from nn.layers.kan_convolutional.KANConv import KAN_Convolutional_Layer
from nn.layers.kan_convolutional.KANLinear import KANLinear

from scipy.io import savemat


class KKAN_Convolutional_Network(nn.Module):
    def __init__(self, device: str = 'cpu'):
        super().__init__()

        self.conv_kan = KAN_Convolutional_Layer(
            n_convs=16,
            kernel_size=(3, 3),
            padding=(1, 1),
            device=device
        )

        self.restore = torch.nn.Conv2d(16, 1, 3, padding=1)

    def forward(self, x):
        x = self.conv_kan(x)
        x = self.restore(x)

        return x


def parse_args():
    parser = argparse.ArgumentParser(description="KAN TRAIN command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")

    parser.add_argument("--db_list", type=str, default="",
                        help="path to train *.txt dataset-filelist.", metavar="")
    parser.add_argument("--db_basedir", type=str, default="",
                        help="basedir for db_list.", metavar="")

    parser.add_argument("--batch_size", type=int, default=20,
                        help="batch size, feeding into the model.", metavar="")

    parser.add_argument("--epoch_number", type=int, default=1,
                        help="number of training epochs.", metavar="")
    parser.add_argument("--learning_rate", type=float, default=8e-3,
                        help="learning rate value.", metavar="")

    return parser.parse_args()


def train(model, device, train_loader, optimizer, criterion):

    model.to(device)
    model.train()
    train_loss = 0
    train_writer = SummaryWriter(log_dir=os.path.join("./runs", datetime.now().strftime("%b%d_%H-%M-%S")))
    # Process the images in batches
    for batch_idx, blob in enumerate(tqdm(train_loader)):
        # Recall that GPU is optimized for the operations we are dealing with
        data, target = blob["x"], blob["y"]
        data, target = data.to(device), target.to(device)
        # Reset the optimizer
        optimizer.zero_grad()

        # Push the data forward through the model layers
        output = model(data)

        # Get the loss
        loss = criterion(output, target)

        # Keep a running total
        train_loss += loss.item()
        if batch_idx % 10 == 0:
            train_writer.add_scalar("batch/loss", loss.item(), global_step=batch_idx)
        if batch_idx % 100 == 0:
            torch.save(model.state_dict(), f'model_step_{batch_idx}.pth')

        # Backpropagate
        loss.backward()
        optimizer.step()

    # return average loss for the epoch
    avg_loss = train_loss / (batch_idx + 1)
    print('Training set: Average loss: {:.6f}'.format(avg_loss))
    return avg_loss



def main(args):
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    # Create dataset.
    trainset = MatDataset(csv_file=args.db_list,
                          root_dir=args.db_basedir)
    trainloader = DataLoader(trainset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             pin_memory=True)

    # Create FNO model.
    model = KKAN_Convolutional_Network(device=device)
    model = model.to(device)

    # Create optimizer.
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=1e-4)

    # Create loss.
    criterion = torch.nn.MSELoss()

    # Train.
    epoch_loss = train(model, device, trainloader, optimizer, criterion)

    # Test.
    # model.eval()
    # model.load_state_dict(torch.load("./model_step_100.pth"))
    # with torch.no_grad():
    #     for batch_idx, blob in enumerate(tqdm(trainloader)):
    #         # Recall that GPU is optimized for the operations we are dealing with
    #         data, target = blob["x"], blob["y"]
    #         data, target = data.to(device), target.to(device)
    #         # Push the data forward through the model layers
    #         output = model(data)
    #
    #         x = data.detach().cpu().numpy()[0].transpose((1, 2, 0)).astype(np.float32)[:,:,0]
    #         y = target.detach().cpu().numpy()[0].transpose((1, 2, 0)).astype(np.float32)[:,:,0]
    #         pred = output.detach().cpu().numpy()[0].transpose((1, 2, 0)).astype(np.float32)[:,:,0]
    #         residual = np.abs(pred - y)
    #         residual = (residual - np.amin(residual)) / (np.amax(residual) - np.amin(residual))
    #         residual_target = np.abs(x - y)
    #         residual_target = (residual_target - np.amin(residual_target)) / (np.amax(residual_target) - np.amin(residual_target))
    #
    #         savemat(os.path.join("pred_mat", "{}.mat".format(batch_idx)),
    #                 {"image": pred})
    #         savemat(os.path.join("x_mat", "{}.mat".format(batch_idx)),
    #                 {"image": x})
    #         savemat(os.path.join("y_mat", "{}.mat".format(batch_idx)),
    #                 {"image": y})

            # cv2.imwrite(os.path.join("./out/sketch", "{}.png".format(batch_idx)), (x * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join("./out/gt", "{}.png".format(batch_idx)), (y * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join("./out/pred", "{}.png".format(batch_idx)), (pred * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join("./out/residual", "{}.png".format(batch_idx)), (residual * 255).astype(np.uint8))
            # cv2.imwrite(os.path.join("./out/residual_target", "{}.png".format(batch_idx)), (residual_target * 255).astype(np.uint8))

    # Save.
    torch.save(model.state_dict(), 'model_weights.pth')


if __name__ == "__main__":
    print("PyTorch version: {}".format(torch.__version__))
    main(parse_args())
