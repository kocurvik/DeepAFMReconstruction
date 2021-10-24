import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        self.split = split
        self.split_path = os.path.join(path, "{}.npy".format(split))

        print("Loading dataset from path: ", self.split_path)

        self.entries = torch.from_numpy(np.load(self.split_path))

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.entries)

    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """
        return {'input': self.entries[index, :2], 'gt': self.entries[index, 2]}


if __name__ == '__main__':
    path = 'D:/Research/data/GEFSEM/synth/generated/'
    dataset = Dataset(path, 'test')
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        cv2.imshow("L", item['input'][0, 0].numpy())
        cv2.imshow("R", item['input'][0, 1].numpy())
        cv2.imshow("GT", item['gt'][0].numpy())
        cv2.waitKey(0)
