import json
import os

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from synth.artifacts import Artifactor
from synth.synthetizer import Synthesizer

if __name__ == '__main__':
    syn = Synthesizer(tips_path='D:/Research/data/GEFSEM/synth/res/tips.pkl')

    art = Artifactor(shadows_prob=0.0, noise_prob=0.0, skew_prob=0.0, overshoot_prob=1.0)
    # art_single = Artifactor(shadows_prob=1.0, noise_prob=0.0, skew_prob=0.0, overshoot_prob=0.0, shadows_uniform_both_p=0.0, shadows_uniform_p=1.0, shadows_randomize_prob=0.0)
    # art_both_random = Artifactor(shadows_prob=1.0, noise_prob=0.0, skew_prob=0.0, overshoot_prob=0.0, shadows_uniform_both_p=1.0, shadows_uniform_p=0.0, shadows_randomize_prob=1.0)
    # art_single_random = Artifactor(shadows_prob=1.0, noise_prob=0.0, skew_prob=0.0, overshoot_prob=0.0, shadows_uniform_both_p=0.0, shadows_uniform_p=0.0, shadows_randomize_prob=1.0)

    while True:

        entry = syn.generate_single_entry(apply_artifacts=False)
        img = entry[0]
        img_l_both, img_r_both = art.apply(img)

        cv2.imshow("IMG", img)
        cv2.imshow("L both", img_l_both)
        cv2.imshow("R both", img_r_both)
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
            cv2.imwrite("figs/overshoot/base.png", 255 * img)
            cv2.imwrite("figs/overshoot/L.png", 255 * img_l_both)
            cv2.imwrite("figs/overshoot/R.png", 255 * img_r_both)
            # cv2.imwrite("figs/shadows/L_single.png", 255 * img_l_single)
            # cv2.imwrite("figs/shadows/R_single.png", 255 * img_r_single)
            # cv2.imwrite("figs/shadows/L_both_random.png", 255 * img_l_both_random)
            # cv2.imwrite("figs/shadows/R_both_random.png", 255 * img_r_both_random)
            # cv2.imwrite("figs/shadows/L_single_random.png", 255 * img_l_single_random)
            # cv2.imwrite("figs/shadows/R_single_random.png", 255 * img_r_single_random)

