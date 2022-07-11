import argparse
import itertools
import json
import os
import pickle

import numpy as np
import cv2
import scipy
import torch
from PIL import Image

from eval.registration import register_affine_orb, register_rigid_sitk, resample_images
from network.train import load_model
from network.unet import ResUnet
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true', default=False)
    parser.add_argument('-i', '--images', action='store_true', default=False)
    parser.add_argument('-ai', '--aligned_images', action='store_true', default=False)
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args


def baseline_lr_filtering(img_1, img_2, threshold=0.01):
    """
    Base solution to "shadow" problem. Same as in NN source codes.
    :param img_1: first (L or R) image
    :param img_2: second (L or R) image
    :param threshold: thresholding of diff
    :return: result of base filter
    """
    img_1 = np.squeeze(img_1)
    img_2 = np.squeeze(img_2)

    diff = img_1 - img_2

    res = (img_1 + img_1) / 2
    res = np.where(diff < -threshold, img_1, res)
    res = np.where(diff > threshold, img_2, res)
    return res

def apply_baseline(entries):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        img_baseline = normalize(baseline_lr_filtering(img_l_normalized, img_r_normalized))
        entry['img_out'] = denormalize(img_baseline, [img_l, img_r])
    return entries


def output_aligned_images(entries):
    pdf_imgs = []

    entry_first = entries[0]
    images = [Image.fromarray((255 * normalize(entry_first[x])).astype(np.uint8)) for x in ['img_l', 'img_r', 'img_out']]
    img_t = 255 * normalize(entry_first['img_out'])
    cv2.imwrite("figs/aligned/{}_out.png".format(0), img_t)

    for idx in range(1, len(entries)):
        images = []
        entry = entries[idx]
        # for img_key in ['img_l', 'img_r', 'img_out']:
        for img_key in ['img_out']:
            orig_img = entry_first[img_key]
            t_img = entry[img_key]
            transform, metric_value = register_rigid_sitk(orig_img, t_img, metric='mse', p1=entry_first['registration_points'], p2=entry['registration_points'], verbose=False)
            img_t, _ = resample_images(normalize(orig_img), normalize(t_img), transform)
            images.append(255 * normalize(img_t).astype(np.uint8))
            cv2.imwrite("figs/aligned/{}_out.png".format(idx), 255 * normalize(img_t))

def inference(model, entries):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        nn_input = torch.from_numpy(np.stack([img_l_normalized, img_r_normalized], axis=0)[None, ...]).float().cuda()
        img_nn = model(nn_input).detach().cpu().numpy()[0, 0, ...]
        entry['img_out_normalized'] = img_nn
        entry['img_out'] = denormalize(img_nn, [img_l, img_r])
    return entries


def main():
    entries_path = os.path.join('D:/Research/data/GEFSEM/EvalData/Kremik/entries.pkl')
    print("Loading data from path: ", entries_path)
    with open(entries_path, 'rb') as f:
        entries = pickle.load(f)

    # if args.model_path == 'baseline':
    # entries = apply_baseline(entries)

    model = ResUnet(2).cuda()
    model.load_state_dict(torch.load('checkpoints/b7cf_early.pth'))
    model.eval()


    entries = inference(model, entries)
    # else:
        # model = ResUnet(2).cuda()
        # print("Resuming from: ", args.model_path)
        # model_basename = os.path.basename(args.model_path).split('.')[0]
        # model.load_state_dict(torch.load(args.model_path))
        # model.eval()
        # list_of_entries = [inference(model, entries) for entries in list_of_entries]

    output_aligned_images(entries)



if __name__ == '__main__':
    main()