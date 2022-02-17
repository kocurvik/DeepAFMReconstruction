import json
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from synth.artifacts import Artifactor
from synth.synthetizer import Synthesizer
from utils.image import normalize_joint, subtract_mean_plane_both


class OnlineDataset(torch.utils.data.Dataset):
    def __init__(self, path, num_items=25000):
        json_path = os.path.join(path, "{}.json".format('val'))
        print("Loading config from val data: ", json_path)
        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        json_dict['args']['num_items'] = num_items
        self.num_items = num_items

        json_dict['args']['apply_artifacts'] = True

        self.synthesizer = Synthesizer(**json_dict['args'])

    def __len__(self):
        return self.num_items

    def __getitem__(self, item):
        entry = self.synthesizer.generate_single_entry(apply_artifacts=True)
        return {'input': torch.from_numpy(entry[:2]), 'gt':torch.from_numpy(entry[2])}


class PregeneratedDataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        self.split = split
        self.split_path = os.path.join(path, "{}.npy".format(split))

        self.subtract_mean_plane = False

        json_path = os.path.join(path, "{}.json".format(split))
        print("Data without artifacts loading config from: ", json_path)
        with open(json_path, 'r') as f:
            json_dict = json.load(f)

        print("Loading dataset from path: ", self.split_path)
        if json_dict['args']['apply_artifacts']:
            self.apply_artifacts = False
            self.entries = torch.from_numpy(np.load(self.split_path))
            print("Artifacts already applied")
        else:
            self.apply_artifacts = True
            self.artifactor = Artifactor(**json_dict["synthetizer_params"])
            self.entries = np.load(self.split_path)
            if json_dict['args']['subtract_mean_plane']:
                self.subtract_mean_plane = True
            print("Artifacts will be applied on the run")


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
        if self.apply_artifacts:
            image_dil = self.entries[index, 0]
            image_gt = self.entries[index, 1]
            image_l, image_r = self.artifactor.apply(image_dil)
            if self.subtract_mean_plane:
                image_l, image_r, plane = subtract_mean_plane_both(image_l, image_r,return_plane=True)
                image_gt -= plane
            images = normalize_joint([image_l, image_r, image_gt])
            entry = torch.from_numpy(np.stack(images, axis=0).astype(np.float32))
        else:
            entry = self.entries[index]

        return {'input': entry[:2], 'gt': entry[2]}





if __name__ == '__main__':
    path = 'D:/Research/data/GEFSEM/synth/generated/555e6565715b5fd76a38f56c7fbf2098ab3e69a3/'
    dataset = OnlineDataset(path)
    # dataset.artifactor.shadows_prob = 1.0
    # dataset.artifactor.overshoot_prob = 0.0
    # dataset.artifactor.noise_prob = 0.0
    # dataset.artifactor.skew_prob = 0.0
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for item in data_loader:
        cv2.imshow("L", item['input'][0, 0].numpy())
        cv2.imshow("R", item['input'][0, 1].numpy())
        cv2.imshow("GT", item['gt'][0].numpy())
        cv2.waitKey(1)

        plt.plot(item['input'][0, 0].numpy()[64, :], 'r')
        plt.plot(item['input'][0, 1].numpy()[64, :], 'b')
        plt.plot(item['gt'][0].numpy()[64, :])
        plt.show()

