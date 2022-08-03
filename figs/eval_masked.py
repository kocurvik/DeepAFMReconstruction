import argparse
import itertools
import json
import os
import pickle
from copy import copy, deepcopy
from multiprocessing import Pool

import numpy as np
import cv2
import scipy
import torch
from PIL import Image

from eval.registration import register_rigid_sitk, resample_images
from eval.run_eval import apply_baseline, inference
from network.train import load_model
from network.unet import ResUnet
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize, \
    subtract_mean_plane, subtract_mean_plane_both, line_by_line_level


def get_line(entries, model, level, ll, use_mask, i=1):
    entry_base = apply_baseline(deepcopy(entries), gauss=False, average=True, median=False, threshold=0.0, level=level, ll=ll, use_mask=use_mask)[0]
    entry_net = inference(model, deepcopy(entries), level=level, ll=ll, use_mask=use_mask)[0]

    return [entry_base['img_l_level'], entry_base['img_r_level'], entry_base['img_out'], entry_net['img_out']]

def main():
    model = ResUnet(2).cuda()
    model_path = 'checkpoints/555e_earliest.pth'
    print("Resuming from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    list_of_entries = []
    entries_path = 'D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/EvalMasked/Neno/entries.pkl'
    print("Loading data from path: ", entries_path)
    data_basename = dir
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)

    e = 2
    entries = entries[e:e+1]

    imgs = []

    imgs.extend(get_line(entries, model, level=True, ll=0, use_mask=False))
    imgs.extend(get_line(entries, model, level=True, ll=0, use_mask=True))
    imgs.extend(get_line(entries, model, level=False, ll=1, use_mask=True))

    normalized_imgs = normalize_joint(imgs)

    level_names = ['mps', 'mps_mask', 'll1_mask']
    level_names_full = ['Mean Plane Subtraction \\\\ No Mask', 'Mean Plane Subtraction \\\\ With Mask', 'Linear Line-by-Line Fitting \\\\ With Mask']
    img_names = ['l', 'r', 'base', 'net']

    print('\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}m{#1}}'
          '\\begin{tabular}{C{0.05\\textwidth} C{0.20\\textwidth} C{0.20\\textwidth} C{0.20\\textwidth} C{0.20\\textwidth}}'
          '~ & L$\\rightarrow$R (After Leveling) & R$\\rightarrow$L (After Leveling) & Baseline + Avg. Filter & ResU-Net (ours) \\\\')

    for i in range(3):
        print('\\rotatebox{90}{\\makecell{' + level_names_full[i] + '}} &')
        for j in range(4):
            img_path = '{}_{}.png'.format(level_names[i], img_names[j])
            print('\\includegraphics[width=0.20\\textwidth]{images/eval_masked/' + img_path + '}')
            if j == 3:
                print('\\\\')
            else:
                print('&')
            cv2.imwrite('figs/masked_eval/' + img_path, (255 * normalized_imgs[j + 4 * i]).astype(np.uint8))

    print('\\end{tabular}')

if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    main()