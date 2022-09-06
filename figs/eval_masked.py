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
    model_path = 'checkpoints/555e_mreg_wgl_009.pth'
    print("Resuming from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    list_of_entries = []
    entries_path_neno = 'D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/EvalMasked/Neno/entries.pkl'
    print("Loading data from path: ", entries_path_neno)
    data_basename = dir
    with open(entries_path_neno, 'rb') as f:
        entries_neno = pickle.load(f)

    e = 0
    entries_neno = entries_neno[e:e+1]

    imgs = []

    imgs.extend(normalize_joint(get_line(entries_neno, model, level=True, ll=0, use_mask=False)))
    imgs.extend(normalize_joint(get_line(entries_neno, model, level=True, ll=0, use_mask=True)))
    imgs.extend(normalize_joint(get_line(entries_neno, model, level=False, ll=1, use_mask=True)))

    normalized_imgs = normalize_joint(imgs)


    entries_path_logos = 'D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/EvalMasked/LogosRot/entries.pkl'
    print("Loading data from path: ", entries_path_logos)
    data_basename = dir
    with open(entries_path_logos, 'rb') as f:
        entries_logos = pickle.load(f)

    e = 2
    entries_logos = entries_logos[e:e+1]

    imgs.extend(normalize_joint(get_line(entries_logos, model, level=True, ll=0, use_mask=False)))
    imgs.extend(normalize_joint(get_line(entries_logos, model, level=True, ll=0, use_mask=True)))
    imgs.extend(normalize_joint(get_line(entries_logos, model, level=False, ll=1, use_mask=True)))



    level_names = ['mps', 'mps_mask', 'll1_mask']
    sample_names = ['neno', 'logosrot']
    level_names_full = ['MPS', 'MPS + Mask', 'LBLF + Mask']
    img_names = ['l', 'r', 'base', 'net']

    print('\\newcolumntype{C}[1]{>{\\centering\\arraybackslash}m{#1}}')

    print('\\begin{tabular}{C{0.03\\textwidth} C{0.18\\textwidth} C{0.18\\textwidth} C{0.18\\textwidth} C{0.18\\textwidth}}'
          '~ & L$\\rightarrow$R (After Leveling) & R$\\rightarrow$L (After Leveling) & Baseline + Avg. Filter & ResU-Net (ours) \\\\')

    for k in range(2):
        for i in range(3):
            print('\\rotatebox{90}{\\makecell{' + level_names_full[i] + '}} &')
            for j in range(4):
                img_path = '{}_{}_{}.png'.format(sample_names[k], level_names[i], img_names[j])
                print('\\includegraphics[width=0.18\\textwidth]{images/eval_masked/' + img_path + '}')
                if j == 3:
                    print('\\\\')
                else:
                    print('&')
                cv2.imwrite('figs/masked_eval/' + img_path, (255 * imgs[k * 12 + j + 4 * i]).astype(np.uint8))

    print('\\end{tabular}')

if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    main()