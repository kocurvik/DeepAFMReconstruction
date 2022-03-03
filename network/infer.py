import argparse

import cv2
import torch
from torch.utils.data import DataLoader

from network.dataset import PregeneratedDataset
from network.unet import ResUnet


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subset', type=str, default='val', help='Which subset of the dataset to use')
    parser.add_argument('model_path', help='Path to the model .pth file')
    parser.add_argument('data_path', help='Path to the dataset folder containing the json and npy files')
    args = parser.parse_args()
    return args


def infer(args):
    # Show the results on a dataset provided.
    model = ResUnet(2).cuda()
    print("Resuming from: ", args.model_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    dataset = PregeneratedDataset(args.data_path, args.subset)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for sample in loader:
        pred = model(sample['input'].cuda())[0, 0, :, :].detach().cpu().numpy()
        gt = sample['gt'][0, :, :].numpy()
        img_l = sample['input'][0, 0, :, :].numpy()
        img_r = sample['input'][0, 1, :, :].numpy()

        cv2.imshow("img_l", cv2.resize(img_l, (512, 512)))
        cv2.imshow("img_r", cv2.resize(img_r, (512, 512)))
        cv2.imshow("gt", cv2.resize(gt, (512, 512)))
        cv2.imshow("pred", cv2.resize(pred, (512, 512)))
        cv2.waitKey(0)

if __name__ == '__main__':
    args = parse_command_line()
    infer(args)