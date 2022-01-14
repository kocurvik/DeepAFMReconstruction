import argparse
import datetime
import json
import time

import git
import torch
import numpy as np
import os

from network.dataset import PregeneratedDataset, OnlineDataset
from torch.utils.data import DataLoader

from network.unet import ResUnetPlusPlus, ResUnet


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-r', '--resume', type=int, default=None, help='checkpoint to resume from')
    parser.add_argument('-nw', '--workers', type=int, default=0, help='workers')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--no_preload', action='store_true', default=False)
    parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    parser.add_argument('-exp', type=int, default=0, help='number of experiment')
    parser.add_argument('-expr', type=int, default=0, help='number of experiment to resume from')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    parser.add_argument('-o', '--opt', type=str, default='sgd', help='optimizer to use: adam or sgd')
    parser.add_argument('-l', '--loss', type=str, default='l2', help='loss to use l1 or l2')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0, help='weight decay to use during training')
    parser.add_argument('-wgl', '--weight_grad_loss', type=float, default=0.0, help='weight of the grad part of the loss')
    parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('path')
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args



def load_model(args):
    """ Loads model. If args.resum is None weights for the backbone are pre-trained on ImageNet, otherwise previous
    checkpoint is loaded """

    data_dir_name = os.path.basename(os.path.normpath(args.path))
    exp_dir = os.path.join('checkpoints', data_dir_name, '{:03d}'.format(args.exp))
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)

    model = ResUnet(2).cuda()
    if args.resume is not None:
        resume_path = os.path.join('checkpoints', data_dir_name, '{:03d}'.format(args.expr), '{:03d}.pth'.format(args.resume))
        print("Resuming from: ", resume_path)
        model.load_state_dict(torch.load(resume_path))
    return model, exp_dir


def train(args):
    model, save_dir = load_model(args)

    train_dataset = OnlineDataset(args.path, num_items=16)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    val_dataset = PregeneratedDataset(args.path, 'val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print("Optimizer: ", args.opt)
    print("Learning rate: ", args.learning_rate)
    print("Weight decay: ", args.weight_decay)


    train_loss_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()

    if args.loss == 'l2':
        print("Using L2 loss")
        loss_layer = torch.nn.MSELoss()
    elif args.loss == 'l1':
        print("Using L1 loss")
        loss_layer = torch.nn.L1Loss()
    else:
        print("Using L2 loss")
        loss_layer = torch.nn.MSELoss()

    if args.weight_grad_loss > 0.0:
        print("Using grad loss with weight: ", args.weight_grad_loss)
        sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_xy = np.stack([sobel_y.T, sobel_y], axis=0)
        sobel_xy_kernel = torch.tensor(sobel_xy, dtype=torch.float32).unsqueeze(1).cuda()
        train_loss_direct_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()
        train_loss_grad_running = torch.from_numpy(np.array([0], dtype=np.float32)).cuda()

    start_epoch = 0 if args.resume is None else args.resume + 1

    repo = git.Repo(search_parent_directories=True)
    param_json_dict = {'date': datetime.datetime.today().strftime('%Y-%m-%d-%H:%M:%S'), 'git_hash': repo.head.object.hexsha, 'args': vars(args)}
    param_json_path = os.path.join(save_dir, 'train_params.json')
    with open(param_json_path,'w') as f:
        json.dump(param_json_dict, f, sort_keys=True, indent=4)
    print("Dumped params to: ", param_json_path)

    epoch_csv_path = os.path.join(save_dir, 'train_vals.csv')
    with open(epoch_csv_path, 'w') as f:
        if args.weight_grad_loss > 0.0:
            f.write('epoch, train_loss_avg, train_loss_running, train_loss_running_direct, train_loss_running_grad, val_loss_avg, val_loss_avg_direct, val_loss_avg_grad \n')
        else:
            f.write('epoch, train_loss_avg, train_loss_running, val_loss_avg \n')

    print("Starting at epoch {}".format(start_epoch))
    print("Running till epoch {}".format(args.epochs))

    for e in range(start_epoch, args.epochs):
        print("Starting epoch: ", e)
        epoch_start_time = time.time()
        train_loss_avg = 0.0
        for i, sample in enumerate(train_loader):
            pred = model(sample['input'].cuda())[:, 0, :, :]
            gt = sample['gt'].cuda()
            optimizer.zero_grad()

            loss = loss_layer(pred, gt)

            if args.weight_grad_loss > 0.0:
                pred_grad_xy = torch.nn.functional.conv2d(pred.unsqueeze(1), sobel_xy_kernel)
                gt_grad_xy = torch.nn.functional.conv2d(gt.unsqueeze(1), sobel_xy_kernel)
                grad_loss = args.weight_grad_loss * loss_layer(pred_grad_xy, gt_grad_xy)

                train_loss_direct_running = 0.9 * train_loss_direct_running + 0.1 * loss
                train_loss_grad_running = 0.9 * train_loss_grad_running + 0.1 * grad_loss

                loss += grad_loss

            train_loss_running = 0.9 * train_loss_running + 0.1 * loss
            train_loss_avg += loss.item()

            remaining_time = (time.time() - epoch_start_time) / (i + 1) * (len(train_loader) - i)

            if i % 100 == 0:
                if args.weight_grad_loss > 0.0:
                    print("At step {}/{} - epoch eta: {} - running loss: {} - running direct loss: {} - running grad loss: {}".
                          format(i, len(train_loader), datetime.timedelta(seconds=remaining_time), train_loss_running.item(), train_loss_direct_running.item(), train_loss_grad_running.item()))

                else:
                    print("At step {}/{} - epoch eta: {} - running loss: {}".format(i, len(train_loader), datetime.timedelta(seconds=remaining_time), train_loss_running.item()))


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss_avg /= len(train_loader)
        print("At step {}/{} - running loss: {}".format(i, len(train_loader), train_loss_running.item()))
        print("At step {}/{} - avg loss: {}".format(i, len(train_loader), train_loss_avg))

        with torch.no_grad():
            val_losses = []
            val_losses_direct = []
            val_losses_grad = []

            # val_angles = []
            # val_magnitudes = []

            for sample in val_loader:
                pred = model(sample['input'].cuda())[:, 0, :, :]
                optimizer.zero_grad()
                gt = sample['gt'].cuda()

                loss = loss_layer(pred, gt)

                if args.weight_grad_loss > 0.0:
                    pred_grad_xy = torch.nn.functional.conv2d(pred.unsqueeze(1), sobel_xy_kernel)
                    gt_grad_xy = torch.nn.functional.conv2d(gt.unsqueeze(1), sobel_xy_kernel)
                    grad_loss = args.weight_grad_loss * loss_layer(pred_grad_xy, gt_grad_xy)

                    val_losses_direct.append(loss.item())
                    val_losses_grad.append(grad_loss.item())

                    loss += grad_loss

                val_losses.append(loss.item())

            print(20 * "*")
            print("Epoch {}/{}".format(e, args.epochs))
            print("val loss: {}".format(np.mean(val_losses)))
            if args.weight_grad_loss > 0.0:
                print("val loss direct: {}".format(np.mean(val_losses_direct)))
                print("val loss grad: {}".format(np.mean(val_losses_grad)))
            print(20 * "*")

        if args.weight_grad_loss > 0.0:
            epoch_csv_line = '{:03d}, {:.8e}, {:.8e}, {:.8e}, {:.8e}, {:.8e}, {:.8e}, {:.8e} \n'.format(
                e, train_loss_avg, train_loss_running.item(), train_loss_direct_running.item(), train_loss_grad_running.item(), np.mean(val_losses), np.mean(val_losses_direct), np.mean(val_losses_grad))
        else:
            epoch_csv_line = '{:03d}, {:.8e}, {:.8e}, {:.8e} \n'.format(e, train_loss_avg, train_loss_running.item(), np.mean(val_losses))
        with open(epoch_csv_path,'a') as f:
            f.write(epoch_csv_line)

        if args.dump_every != 0 and (e) % args.dump_every == 0:
            print("Saving checkpoint")
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, '{:03d}.pth'.format(e)))


if __name__ == '__main__':
    args = parse_command_line()
    train(args)
