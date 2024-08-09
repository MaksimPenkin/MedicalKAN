# """
# @author   Maksim Penkin
# """

import os
import argparse
from argparse import RawTextHelpFormatter

from nn import models
from data.ixi import ixi

from utils.torch_utils import train_func


def parse_args():
    parser = argparse.ArgumentParser(description="Command-line arguments", usage="%(prog)s [-h]", formatter_class=RawTextHelpFormatter)

    parser.add_argument("--use_gpu", type=int, default=0,
                        help="gpu index to be used.", metavar="")

    parser.add_argument("--db", type=str, default="",
                        help="path to the dataset-filelist.", metavar="")
    parser.add_argument("--db_basedir", type=str, default="",
                        help="basedir for paths inside the dataset-filelist.", metavar="")

    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size.", metavar="")
    parser.add_argument("--epochs", type=int, default=1,
                        help="number of epochs.", metavar="")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate value.", metavar="")

    return parser.parse_args()


def main(args):
    device = "cuda" if args.use_gpu >= 0 else "cpu"
    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.use_gpu)

    # Dataset.
    dataloader = ixi(args.db,
                     root=args.db_basedir,
                     keys=("sketch", "image"),
                     batch_size=args.batch_size,
                     shuffle=True,
                     pin_memory=True)

    # Model.
    model = models.get(args.nn)

    # Train!
    train_func(model, dataloader,
               criterion=args.loss,
               optimizer=args.optimizer,
               callbacks=None,
               epochs=args.epochs,
               val_dataloader=None,
               device=device)


if __name__ == "__main__":
    main(parse_args())



#######################3

    # Create optimizer.
    # optimizer = torch.optim.Adam(model.parameters(),
    #                              lr=args.learning_rate,
    #                              weight_decay=1e-4)

    # Create loss.
    # criterion = torch.nn.MSELoss()

    # Train.
    # epoch_loss = train(model, device, trainloader, optimizer, criterion, epochs=args.epoch_number)

    # # Test.
    # model.eval()
    # model.load_state_dict(torch.load("./model_e1_s1100.pth"))
    # with torch.no_grad():
    #     for batch_idx, (x, y) in enumerate(tqdm(trainloader)):
    #         # Recall that GPU is optimized for the operations we are dealing with
    #         data, target = x, y
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
    #
    #         cv2.imwrite(os.path.join("./out/sketch", "{}.png".format(batch_idx)), (x * 255).astype(np.uint8))
    #         cv2.imwrite(os.path.join("./out/gt", "{}.png".format(batch_idx)), (y * 255).astype(np.uint8))
    #         cv2.imwrite(os.path.join("./out/pred", "{}.png".format(batch_idx)), (pred * 255).astype(np.uint8))
    #         cv2.imwrite(os.path.join("./out/residual", "{}.png".format(batch_idx)), (residual * 255).astype(np.uint8))
    #         cv2.imwrite(os.path.join("./out/residual_target", "{}.png".format(batch_idx)), (residual_target * 255).astype(np.uint8))

    # Save.
    # torch.save(model.state_dict(), 'model_weights.pth')
