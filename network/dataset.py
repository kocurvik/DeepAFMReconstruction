import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader


def prepare_model_input(img_l, img_r):
    min_height = (min(img_l.shape[0], img_r.shape[0]) // 8) * 8
    min_width = (min(img_l.shape[1], img_r.shape[1]) // 8) * 8

    img_l = img_l[:min_height, :min_width]
    img_r = img_r[:min_height, :min_width]

    img_l_torch = torch.from_numpy(img_l)
    img_r_torch = torch.from_numpy(img_r)

    return torch.stack([img_l_torch, img_r_torch], dim=0)[None, ...], img_l, img_r

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, preload=True):
        self.dataset_dir = os.path.dirname(path)
        self.split = split
        self.preload = preload

        self.split_path = os.path.join(path, split)

        print("Loading dataset from path: ", self.split_path)

        file_list = os.listdir(self.split_path)

        self.filenames = [filename for filename in file_list if '.png' in filename and 'L' not in filename and 'R' not in filename]

        if self.preload:
            print("Preloading data to memory")
            self.entries = []
            for filename in self.filenames:
                entry = self.load_entry(filename)
                self.entries.append(entry)

    def __len__(self):
        """
        Length of dataset
        :return: number of elements in dataset
        """
        return len(self.filenames)

    def load_entry(self, filename):
        gt_img = cv2.cvtColor(cv2.imread(os.path.join(self.split_path, filename)), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

        s = filename.split('.')
        l_img_path = os.path.join(self.split_path, s[0] + 'L.' + s[1])
        l_img = cv2.cvtColor(cv2.imread(l_img_path), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

        r_img_path = os.path.join(self.split_path, s[0] + 'R.' + s[1])
        r_img = cv2.cvtColor(cv2.imread(r_img_path), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255

        entry = {'gt_img': torch.from_numpy(gt_img), 'r_img': torch.from_numpy(r_img), 'l_img': torch.from_numpy(l_img)}
        return entry


    def __getitem__(self, index):
        """
        Returns one sample for training
        :param index: index of entry
        :return: dict containing sample data
        """

        if self.preload:
            entry = self.entries[index]
        else:
            entry = self.load_entry(self.filenames[index])

        input = torch.stack([entry['l_img'], entry['r_img']], dim=0)
        return {'input': input, 'gt': entry['gt_img']}


if __name__ == '__main__':
    path = 'D:/Research/code/old_gefsem/synth/datasets/images/'
    dataset = Dataset(path, 'train', preload=False)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for item in data_loader:
        cv2.imshow("L", item['input'][0, 0].numpy())
        cv2.imshow("R", item['input'][0, 1].numpy())
        cv2.imshow("GT", item['gt'][0].numpy())
        cv2.waitKey(0)


        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # ax.scatter(xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel(), marker='o')

        # plt.show()
