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



def get_metric(img_1, img_2, metric='mse', p1=None, p2=None):
    if metric == 'correlation':
        transform, metric_value = register_rigid_sitk(normalize(img_1), normalize(img_2), metric=metric, p1=p1, p2=p2, verbose=False)
        metric_value = -metric_value
    else:
        transform, metric_value = register_rigid_sitk(img_1, img_2, metric=metric, p1=p1, p2=p2, verbose=False)


    img_2_t, img_c = resample_images(normalize(img_1), normalize(img_2), transform)

    cv2.imshow("Orig img 1", normalize(img_1))
    cv2.imshow("Orig img 2", normalize(img_2))
    cv2.imshow("Transformed img 2", normalize(img_2_t))
    cv2.imshow("Composite img", normalize(img_c))
    key = cv2.waitKey(0)
    if key == ord('s'):
        cv2.imwrite("figs/align_pair/img_1.png", 255 * normalize(img_1))
        cv2.imwrite("figs/align_pair/img_2.png", 255 * normalize(img_2))
        cv2.imwrite("figs/align_pair/img_2_t.png", 255 * normalize(img_2_t))
        cv2.imwrite("figs/align_pair/img_composite.png", 255 * normalize(img_c))

    return metric_value, img_c


def eval_same_sample(entries):
    for idx_1, idx_2 in itertools.combinations(np.arange(len(entries)), 2):
        print("Images: {} and {}".format(entries[idx_1]['filename'], entries[idx_2]['filename']))
        for metric in ['mse', 'correlation']:
            get_metric(entries[idx_1]['img_out'], entries[idx_2]['img_out'], p1=entries[idx_1]['registration_points'], p2=entries[idx_2]['registration_points'], metric=metric)


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


def main(model_path, data_path):
    print("Checking dirs: ", data_path)
    dirs = [dir for dir in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, dir))]

    list_of_entries = []
    for dir in dirs:
        entries_path = os.path.join(data_path, dir, 'entries.pkl')
        print("Loading data from path: ", entries_path)
        with open(entries_path, 'rb') as f:
            entries = pickle.load(f)
        list_of_entries.append(entries)


    model = ResUnet(2).cuda()
    print("Resuming from: ", model_path)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    list_of_entries = [inference(model, entries) for entries in list_of_entries[0:]]

    for entries, dir in zip(list_of_entries, dirs):
        eval_same_sample(entries)


if __name__ == '__main__':
    model_path = 'checkpoints/b7cf_early.pth'
    data_path = 'D:/Research/data/GEFSEM/EvalData'
    main(model_path, data_path)
