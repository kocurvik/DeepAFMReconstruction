import json
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from synth.artifacts import Artifactor
from synth.synthetizer import Synthesizer
from utils.image import normalize, normalize_joint


def linear_skew(sigma_a=0.15, sigma_b=0.15):
    x = np.linspace(0.0, 1.0, img.shape[1])
    y = np.linspace(0.0, 1.0, img.shape[0])
    mg_x, mg_y = np.meshgrid(x, y)
    a = sigma_a * np.random.randn()
    b = sigma_b * np.random.randn()

    return a * mg_x + b * mg_y


def parabolic_skew(sigma_a=0.15, sigma_b=0.15):
    c_x = np.random.rand()
    c_y = np.random.rand()

    a = sigma_a * np.random.randn()
    b = sigma_b * np.random.randn()

    x = np.linspace(0, 1.0, img.shape[1])
    y = np.linspace(0, 1.0, img.shape[0])
    mg_x, mg_y = np.meshgrid(x, y)

    return a * (mg_x - c_x) ** 2 + b * (mg_y - c_y) ** 2


def save_fig(image, filename):
    plt.plot(image[64, :])
    plt.ylim([0, 1])
    plt.margins(x=0)
    plt.savefig(filename, bbox_inches='tight')
    # plt.show()
    plt.cla()



if __name__ == '__main__':
    syn = Synthesizer(tips_path='D:/Research/data/GEFSEM/synth/res/tips.pkl')

    while True:

        entry = syn.generate_single_entry(apply_artifacts=False)
        img = entry[0]

        skew = linear_skew() + parabolic_skew()

        skewed_img = img + skew

        img, skew, skewed_img = normalize_joint([img, skew, skewed_img])

        cv2.imshow("IMG", img)
        cv2.imshow("Skew", normalize(skew))
        cv2.imshow("Skewed img", skewed_img)
        # cv2.imshow("L both", img_l_both)
        # cv2.imshow("R both", img_r_both)
        # cv2.imshow("L single", img_l_single)
        # cv2.imshow("R single", img_r_single)
        # cv2.imshow("L both random", img_l_both_random)
        # cv2.imshow("R both random", img_r_both_random)
        # cv2.imshow("L single random", img_l_single_random)
        # cv2.imshow("R single random", img_r_single_random)

        key = cv2.waitKey(0)
        if key == ord('s'):
        #     save_fig(item['input'][0, 0].numpy(), 'figs/shadows/plot_l.pdf')
        #     save_fig(item['input'][0, 1].numpy(), 'figs/shadows/plot_r.pdf')
        #     save_fig(item['gt'][0].numpy(), 'figs/shadows/gt.pdf')
        #
            cv2.imwrite('figs/skew/base.png', 255 * img)
            cv2.imwrite("figs/skew/skew.png", 255 * normalize(skew))
            cv2.imwrite("figs/skew/skewed_image.png", 255 * skewed_img)

            save_fig(img, 'figs/skew/plot_base.pdf')
            save_fig(skew, 'figs/skew/plot_skew.pdf')
            save_fig(skewed_img, 'figs/skew/skewed_img.pdf')

        # cv2.imwrite("figs/shadows/L_both.png", 255 * img_l_both)
            # cv2.imwrite("figs/shadows/R_both.png", 255 * img_r_both)
            # cv2.imwrite("figs/shadows/L_single.png", 255 * img_l_single)
            # cv2.imwrite("figs/shadows/R_single.png", 255 * img_r_single)
            # cv2.imwrite("figs/shadows/L_both_random.png", 255 * img_l_both_random)
            # cv2.imwrite("figs/shadows/R_both_random.png", 255 * img_r_both_random)
            # cv2.imwrite("figs/shadows/L_single_random.png", 255 * img_l_single_random)
            # cv2.imwrite("figs/shadows/R_single_random.png", 255 * img_r_single_random)

