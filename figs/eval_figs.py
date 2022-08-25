import argparse
import copy
import itertools
import json
import os
import pickle

import numpy as np
import cv2
import scipy
import torch
from PIL import Image

from eval.run_eval import apply_baseline, inference, get_metric, eval_same_sample
from network.train import load_model
from network.unet import ResUnet
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize, \
    subtract_mean_plane, subtract_mean_plane_both

DIRNAME_DICT = {'d008': 'Wafers', 'D010_Bunky': 'Cells', 'INCHAR (MFM sample)': 'Permalloy', 'Kremik': 'Silicon',
                'loga': 'Logos', 'Neno': 'Neno', 'Tescan sample': 'Patterns', 'TGQ1': 'TGQ1', 'TGZ3': 'TGZ3', 'SiliconRot': 'SiliconRot'}

def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--eval', action='store_true', default=False)
    parser.add_argument('-i', '--images', action='store_true', default=False)
    parser.add_argument('-ai', '--aligned_images', action='store_true', default=False)
    parser.add_argument('-g', '--gauss', action='store_true', default=False)
    parser.add_argument('-a', '--average', action='store_true', default=False)
    parser.add_argument('-m', '--median', action='store_true', default=False)
    parser.add_argument('-l', '--level', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float, default=0.01)
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args

def get_multi_input(model, img_l, img_r, tile_size=8):
    img_l, img_r = enforce_img_size_for_nn(img_l, img_r, dim=tile_size)
    stack = np.stack([img_l, img_r], axis=0)

    w_tiles = ((img_l.shape[1] - 128) // tile_size) + 1
    h_tiles = ((img_l.shape[0] - 128) // tile_size) + 1

    output = np.zeros_like(img_l)
    output_counts = np.zeros_like(img_l)

    for i in range(h_tiles):
        for j in range(w_tiles):
            input = stack[np.newaxis, :, i*tile_size: i*tile_size + 128, j*tile_size: j*tile_size + 128]
            output_tile = model(torch.from_numpy(input).float().cuda()).detach().cpu().numpy()

            output[i*tile_size: i*tile_size + 128, j*tile_size: j*tile_size + 128] += output_tile[0, 0]
            output_counts[i*tile_size: i*tile_size + 128, j*tile_size: j*tile_size + 128] += 1

    return output/output_counts


def compose_images(images):
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    img_new = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        img_new.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    return img_new


def output_images(list_of_entries_baseline, list_of_entries_net):
    pdf_imgs = []

    print('\\begin{tabular}{m{0.02\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth}}')

    for entries_baseline, entries_net in zip(list_of_entries_baseline, list_of_entries_net):
        for entry_baseline, entry_net in zip(entries_baseline, entries_net):
            dirname = os.path.basename(os.path.dirname(entry_baseline['gwy_path']))

            baseline_img, ours_img = normalize_joint([entry_baseline['img_out'], entry_net['img_out']])
            img_l, img_r = normalize_joint([entry_baseline['img_l'], entry_baseline['img_r']])

            cv2.imshow('l', normalize(img_l))
            cv2.imshow('r', normalize(img_r))
            cv2.imshow('out_ours', ours_img)
            cv2.imshow('out_baseline', baseline_img)

            # print('MSE: ', np.sqrt(np.mean((entry_baseline['img_out'] - entry_net['img_out'])**2)))

            key = cv2.waitKey(0)

            if key == ord('s'):

                cv2.imwrite('figs/eval/{}_l.png'.format(dirname), 255 * normalize(img_l))
                cv2.imwrite('figs/eval/{}_r.png'.format(dirname), 255 * normalize(img_r))
                cv2.imwrite('figs/eval/{}_out_ours.png'.format(dirname), 255 * ours_img)
                cv2.imwrite('figs/eval/{}_out_baseline.png'.format(dirname), 255 * baseline_img)

        print('\\rotatebox{90}{' + DIRNAME_DICT[dirname] + '}')
        # print('\\hfill')
        print('&')
        print('\\includegraphics[width=0.24\\textwidth]{{images/eval/{}_l.png}}'.format(dirname))
        # print('\\hfill')
        print('&')
        print('\\includegraphics[width=0.24\\textwidth]{{images/eval/{}_r.png}}'.format(dirname))
        # print('\\hfill')
        print('&')
        print('\\includegraphics[width=0.24\\textwidth]{{images/eval/{}_out_baseline.png}}'.format(dirname))
        # print('\\hfill')
        print('&')
        print('\\includegraphics[width=0.24\\textwidth]{{images/eval/{}_out_ours.png}}'.format(dirname))
        print('\\\\')

    print('\\end{tabular}')



    #     for entry in entries:
    #         images = [Image.fromarray((255 * normalize(entry[x])).astype(np.uint8)) for x in ['img_l', 'img_r', 'img_out']]
    #         img_new = compose_images(images)
    #
    #         pdf_imgs.append(img_new)
    #
    # pdf_imgs[0].save("vis/images_{}.pdf".format(model_basename), "PDF", resolution=100.0, save_all=True, append_images=pdf_imgs[1:])


def main(args):
    print("Checking dirs: ", args.data_path)
    dirs = [dir for dir in sorted(os.listdir(args.data_path)) if os.path.isdir(os.path.join(args.data_path, dir))]

    list_of_entries = []
    for dir in dirs:
        entries_path = os.path.join(args.data_path, dir, 'entries.pkl')
        print("Loading data from path: ", entries_path)
        data_basename = dir
        with open(entries_path, 'rb') as f:
            entries = pickle.load(f)
        list_of_entries.append(entries)

    # if args.model_path == 'baseline':
    list_of_entries_baseline = copy.deepcopy(list_of_entries)
    list_of_entries_baseline = [apply_baseline(entries, args.gauss, args.average, args.median, threshold=args.threshold, level=args.level) for entries in list_of_entries_baseline]
    baseline_model_basename = 'baseline_{}'.format(args.threshold)
    if args.gauss:
        baseline_model_basename += '_gauss'
    if args.average:
        baseline_model_basename += '_average'
    if args.median:
        baseline_model_basename += '_median'
    # else:
    model = ResUnet(2).cuda()
    print("Resuming from: ", args.model_path)
    model_basename = os.path.basename(args.model_path).split('.')[0]
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    list_of_entries_net = copy.deepcopy(list_of_entries)
    list_of_entries_net = [inference(model, entries, level=args.level) for entries in list_of_entries_net]

    if args.level:
        model_basename += '_level'

    print("Baseline model basename: " + baseline_model_basename)
    print("Model basename: " + model_basename)

    output_images(list_of_entries_baseline, list_of_entries_net)

if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    args = parse_command_line()
    main(args)