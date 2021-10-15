import argparse
import datetime
import time

import torch
import pytorch_lightning as pl
import numpy as np
import os

from network.dataset import Dataset
from torch.utils.data import DataLoader

from network.model import ResUnetModel
from network.unet import ResUnetPlusPlus, ResUnet


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-r', '--resume', type=int, default=None, help='checkpoint to resume from')
    parser.add_argument('-nw', '--workers', type=int, default=0, help='workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('path')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args



def load_model(args):
    """ Loads model. If args.resum is None weights for the backbone are pre-trained on ImageNet, otherwise previous
    checkpoint is loaded """
    model = ResUnet(2).cuda()
    if args.resume is not None:
        sd_path = 'checkpoints/{:03d}.pth'.format(args.resume)
        print("Resuming from: ", sd_path)
        model.load_state_dict(torch.load(sd_path))
    return model


def train(args):
    model = load_model(args)

    train_dataset = Dataset(args.path, 'train', preload=not args.no_preload)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = Dataset(args.path, 'val', preload=not args.no_preload)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    model = ResUnetModel()
    trainer = pl.Trainer(max_epochs=10, gpus=1)

    trainer.fit(model, train_loader, val_loader)

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    #
    # loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
    #
    # l2_loss = torch.nn.MSELoss()
    #
    # start_epoch = 0 if args.resume is None else args.resume
    # print("Starting at epoch {}".format(start_epoch))
    # print("Running till epoch {}".format(args.epochs))
    #
    #
    # for e in range(start_epoch, args.epochs):
    #     print("Starting epoch: ", e)
    #     epoch_start_time = time.time()
    #     for i, sample in enumerate(train_loader):
    #         pred = model(sample['input'].cuda())[:, 0, :, :]
    #         optimizer.zero_grad()
    #
    #         loss = l2_loss(pred, sample['gt'].cuda())
    #         loss_running = 0.9 * loss_running + 0.1 * loss
    #
    #         remaining_time = (time.time() - epoch_start_time) / (i + 1) * (len(train_loader) - i)
    #
    #         if i % 10 == 0:
    #             print("At step {}/{} - epoch eta: {} - running loss: {}".format(i, len(train_loader), datetime.timedelta(seconds=remaining_time), loss_running.item()))
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #     print("At step {}/{} - running loss: {}".format(i, len(train_loader), loss_running.item()))
    #
    #     with torch.no_grad():
    #         val_losses = []
    #
    #         # val_angles = []
    #         # val_magnitudes = []
    #
    #         for sample in val_loader:
    #             pred = model(sample['input'].cuda())[:, 0, :, :]
    #             optimizer.zero_grad()
    #
    #             loss = l2_loss(pred, sample['gt'].cuda())
    #
    #             val_losses.append(loss.item())
    #
    #         print(20 * "*")
    #         print("Epoch {}/{}".format(e, args.epochs))
    #         print("val loss: {}".format(np.mean(val_losses)))
    #
    #     if args.dump_every != 0 and (e) % args.dump_every == 0:
    #         print("Saving checkpoint")
    #         if not os.path.isdir('checkpoints/'):
    #             os.mkdir('checkpoints/')
    #         torch.save(model.state_dict(), 'checkpoints/{:03d}.pth'.format(e))


if __name__ == '__main__':
    """
    Example usage: python network/train.py -b 32 -e 10 -de 1 -nw 2 -lr 1e-3 /path/to/MLBinsDataset/EXR/dataset.json
    """
    args = parse_command_line()
    train(args)
