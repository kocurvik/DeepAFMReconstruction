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
from matplotlib import pyplot as plt
import matplotlib

font = {'family' : 'cmtt',
        'size'   : 24}

matplotlib.rc('font', **font)


#
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })

from eval.registration import register_rigid_sitk, resample_images
from eval.run_eval import apply_baseline, inference
from network.train import load_model
from network.unet import ResUnet
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize, \
    subtract_mean_plane, subtract_mean_plane_both, line_by_line_level


def get_line(entries, model, level, ll, use_mask):
    entry_base = apply_baseline(deepcopy(entries), gauss=False, average=False, median=False, threshold=0.0, level=level, ll=ll, use_mask=use_mask)[0]
    entry_base_avg = apply_baseline(deepcopy(entries), gauss=False, average=True, median=False, threshold=0.0, level=level, ll=ll, use_mask=use_mask)[0]
    entry_base_median = apply_baseline(deepcopy(entries), gauss=False, average=False, median=True, threshold=0.0, level=level, ll=ll, use_mask=use_mask)[0]
    entry_base_gauss = apply_baseline(deepcopy(entries), gauss=True, average=False, median=False, threshold=0.0, level=level, ll=ll, use_mask=use_mask)[0]
    entry_net = inference(model, deepcopy(entries), level=level, ll=ll, use_mask=use_mask)[0]

    return entry_net['img_l_level'], entry_net['img_r_level'], entry_base['img_out'], entry_base_avg['img_out'], \
           entry_base_median['img_out'], entry_base_gauss['img_out'], entry_net['img_out']


def plot_and_save(line, name, title):
    xs = np.linspace(0, 10, 512)

    plt.figure(figsize=(6, 6))
    plt.plot(xs[:len(line)], 1e9 * line)
    plt.title(title)
    plt.xlim([0, xs[len(line)]])
    plt.ylim([-300, 450])
    axes = plt.gca()
    axes.yaxis.grid()
    plt.ylabel("Z-axis (nm)")
    plt.xlabel("X-axis ($\mu$m)")
    plt.savefig('figs/eval_plots/{}.pdf'.format(name), bbox_inches = "tight")
    # plt.show()


def main():
    model = ResUnet(2).cuda()
    model_path = 'checkpoints/555e_mreg_wgl_009.pth'
    print("Resuming from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    list_of_entries = []
    entries_path = 'D:/Research/data/GEFSEM/UltraMicroscopy2022Q2/EvalData/TGZ3/entries.pkl'
    print("Loading data from path: ", entries_path)
    data_basename = dir
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)

    e = 4
    entries = entries[e:e+1]

    imgs = []

    lines = np.array(get_line(entries, model, level=True, ll=0, use_mask=False))

    # lines -= np.percentile(lines, 5, axis=(1, 2))[:, np.newaxis, np.newaxis]

    row = 120

    plot_and_save(lines[0, row, :], 'R-L', 'R$\\rightarrow$L')
    plot_and_save(lines[1, row, :], 'L-R', 'L$\\rightarrow$R')
    plot_and_save(lines[2, row, :], 'baseline', 'Baseline')
    plot_and_save(lines[3, row, :], 'baseline_avg', 'Baseline + Average')
    plot_and_save(lines[4, row, :], 'baseline_med', 'Baseline + Median')
    plot_and_save(lines[5, row, :], 'baseline_gauss', 'Baseline + Gauss')
    plot_and_save(lines[6, row, :], 'net', 'ResU-Net (ours)')



if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    main()