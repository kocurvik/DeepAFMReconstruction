import json
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from synth.artifacts import Artifactor
from synth.synthetizer import Synthesizer
from utils.image import normalize_joint


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split):
        self.split = split
        self.split_path = os.path.join(path, "{}.npy".format(split))

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
            print(np.mean(np.abs(image_dil - image_gt)))
            image_l, image_r = self.artifactor.apply(image_dil)
            images = normalize_joint([image_l, image_r, image_gt])
            entry = torch.from_numpy(np.stack(images, axis=0).astype(np.float32))
        else:
            entry = self.entries[index]

        return {'input': entry[:2], 'gt': entry[2]}


def save_fig(image, filename):
    plt.plot(image[64, :])
    plt.ylim([0, 1])
    plt.margins(x=0)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.cla()


if __name__ == '__main__':
    syn = Synthesizer(tips_path='D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/synth/res/tips.pkl')

    for i in range(16):
        image_l, image_r, image_gt = syn.generate_single_entry(apply_artifacts=True)

        cv2.imshow("L", image_l)
        cv2.imshow("R", image_r)
        cv2.imshow("GT", image_gt)

        key = cv2.waitKey(1)
        # if key == ord('s'):
        # save_fig(image_l, 'figs/gen/{:02d}_plot_l.pdf'.format(i))
        # save_fig(image_r, 'figs/gen/{:02d}_plot_r.pdf'.format(i))
        # save_fig(image_gt, 'figs/gen/{:02d}_gt.pdf'.format(i))

        cv2.imwrite('figs/gen/{:02d}_img_l.png'.format(i), 255 * image_l)
        cv2.imwrite('figs/gen/{:02d}_img_r.png'.format(i), 255 * image_r)
        cv2.imwrite('figs/gen/{:02d}_img_gt.png'.format(i), 255 * image_gt)


    # for i in range(8):
    #     for j in range(2):
    #         img_num = i * 2 + j
    #         print('\\includegraphics[width=0.15\\textwidth]{{images/generator_mosaic/{:02}_img_gt.png}}'.format(img_num))
    #         print('\\hfill')
    #         print('\\includegraphics[width=0.15\\textwidth]{{images/generator_mosaic/{:02}_img_l.png}}'.format(img_num))
    #         print('\\hfill')
    #         print('\\includegraphics[width=0.15\\textwidth]{{images/generator_mosaic/{:02}_img_r.png}}'.format(img_num))
    #         if j == 0:
    #             print('\\hfill')
    #         else:
    #             print('\\\\')
    #             print('\\vspace{0.02\\textwidth}')

    for i in range(4):
        for j in range(3):
            img_num = i * 3 + j
            print('\\includegraphics[width=0.1\\textwidth]{{images/generator_mosaic/{:02}_img_gt.png}}'.format(img_num))
            print('\\hfill')
            print('\\includegraphics[width=0.1\\textwidth]{{images/generator_mosaic/{:02}_img_l.png}}'.format(img_num))
            print('\\hfill')
            print('\\includegraphics[width=0.1\\textwidth]{{images/generator_mosaic/{:02}_img_r.png}}'.format(img_num))
            if j == 0 or j == 1:
                print('\\hfill')
            else:
                print('\\\\')
                print('\\vspace{0.0125\\textwidth}')

            # print('\\begin{minipage}{0.15\\textwidth}')
            # print('\\center')
            # print('\\includegraphics[width=0.15\textwidth]{images/generator_mosaic/{:01}_img_l.png} \\\\ a'.format(img_num))
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.15\\textwidth}')
            # print('\\center')
            # print('\\includegraphics[width=1\textwidth]{images/generator_mosaic/{:01}_img_l.png} \\\\ a'.format(img_num))
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.15\\textwidth}')
            # print('\\center')
            # print('\\includegraphics[width=1\textwidth]{images/generator_mosaic/{:01}_img_l.png} \\\\ a'.format(img_num))
            # print('\\end{minipage}')



