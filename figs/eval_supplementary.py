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

from eval.run_eval import preprocess, baseline_lr_filtering, inference
from network.unet import ResUnet
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize, \
    subtract_mean_plane, subtract_mean_plane_both

DIRNAME_DICT = {'d008': 'Wafers', 'D010_Bunky': 'Cells', 'INCHAR (MFM sample)': 'Permalloy', 'Kremik': 'Silicon',
                'loga': 'Logos', 'Neno': 'Neno', 'Tescan sample': 'Patterns', 'TGQ1': 'TGQ1', 'TGZ3': 'TGZ3', 'SiliconRot': 'SiliconRot'}


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--level', action='store_true', default=False)
    parser.add_argument('-t', '--threshold', type=float, default=0.01)
    parser.add_argument('model_path')
    parser.add_argument('data_path')
    args = parser.parse_args()
    return args


def apply_baseline(entries, threshold=0.1, level=False):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        # if 'slow' in entry['filename']:
        #     entry['img_out'] = img_r.astype(np.float32)
        #     continue

        img_l, img_r = preprocess(entry, img_l, img_r, level, ll=0, use_mask=False)

        entry['img_l_level'] = img_l
        entry['img_r_level'] = img_r

        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])

        img_baseline = normalize(baseline_lr_filtering(img_l_normalized, img_r_normalized, threshold=threshold))
        img_baseline_gauss = cv2.GaussianBlur(img_baseline, (5, 5), 0)
        img_baseline_average = cv2.blur(img_baseline, (5, 5))
        img_baseline_median = cv2.medianBlur(img_baseline, 5)

        entry['img_out_baseline'] = denormalize(img_baseline, [img_l, img_r])
        entry['img_out_baseline_gauss'] = denormalize(img_baseline_gauss, [img_l, img_r])
        entry['img_out_baseline_average'] = denormalize(img_baseline_average, [img_l, img_r])
        entry['img_out_baseline_median'] = denormalize(img_baseline_median, [img_l, img_r])
    return entries


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


def output_images(list_of_entries):
    print('\\documentclass{article}')
    print('\\usepackage[margin=0.1cm]{geometry}')
    print('\\usepackage[utf8]{inputenc}')
    print('\\usepackage{graphicx}')
    print('\\usepackage{url}')
    print('\\usepackage{subfig}')
    print('\\usepackage{pgfplots}')

    print('\\usepgfplotslibrary{colormaps}'
          '\\pgfplotsset{colormap={gray}{rgb255=(0,0,0) rgb255=(255,255,255)}}')

    print('\\begin{document}')

    for i, entries in enumerate(list_of_entries):
        for j, entry in enumerate(entries):
            dirname = DIRNAME_DICT[os.path.basename(os.path.dirname(entry['gwy_path']))]

            # baseline_img, ours_img = normalize_joint([entry['img_out_baseline'], entry['img_out']])

            out_dir = 'figs/eval_supplementary/'

            l_png = '{}_{}_l.png'.format(i, j)
            r_png = '{}_{}_r.png'.format(i, j)
            b_avg_png = '{}_{}_b_avg.png'.format(i, j)
            b_med_png = '{}_{}_b_med.png'.format(i, j)
            b_gauss_png = '{}_{}_b_gauss.png'.format(i, j)
            b_png = '{}_{}_b_gauss.png'.format(i, j)
            ours_png = '{}_{}_ours.png'.format(i, j)

            # cv2.imwrite(out_dir + l_png, 255 * normalize(entry['img_l']))
            # cv2.imwrite(out_dir + r_png, 255 * normalize(entry['img_r']))
            # cv2.imwrite(out_dir + b_avg_png, 255 * normalize(entry['img_out_baseline_average']))
            # cv2.imwrite(out_dir + b_med_png, 255 * normalize(entry['img_out_baseline_median']))
            # cv2.imwrite(out_dir + b_gauss_png, 255 * normalize(entry['img_out_baseline_gauss']))
            # cv2.imwrite(out_dir + b_png, 255 * normalize(entry['img_out_baseline']))
            # cv2.imwrite(out_dir + ours_png, 255 * normalize(entry['img_out']))

            lr_list = [entry['img_l'], entry['img_r']]

            lr_min = np.min(np.array(lr_list))
            lr_max = np.max(np.array(lr_list))

            img_l, img_r = normalize_joint(lr_list)

            img_list = [entry['img_out_baseline'], entry['img_out_baseline_average'], entry['img_out_baseline_gauss'], entry['img_out_baseline_median'], entry['img_out']]

            img_min = np.min(np.array(img_list))
            img_max = np.max(np.array(img_list))

            img_b, img_b_avg, img_b_gauss, img_b_med, img_ours = normalize_joint(img_list)



            cv2.imwrite(out_dir + l_png, 255 * normalize(img_l))
            cv2.imwrite(out_dir + r_png, 255 * normalize(img_r))
            cv2.imwrite(out_dir + b_avg_png, 255 * img_b)
            cv2.imwrite(out_dir + b_med_png, 255 * img_b_avg)
            cv2.imwrite(out_dir + b_gauss_png, 255 * img_b_gauss)
            cv2.imwrite(out_dir + b_png, 255 * img_b_med)
            cv2.imwrite(out_dir + ours_png, 255 * img_ours)

            # print('\\vspace*{1em}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + l_png + '} \\\\ L$\\rightarrow$R')
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + r_png + '} \\\\ R$\\rightarrow$L')
            # print('\\end{minipage}')
            # print('\\hfill* \\strut')
            #
            # print('\\\\')
            # print()
            # print('\\vspace{1em}')
            #
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + b_png + '} \\\\ Baseline')
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + b_avg_png + '} \\\\ Baseline + Average')
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + b_gauss_png + '} \\\\ Baseline + Gauss')
            # print('\\end{minipage}')
            # print('\\hfill* \\strut')
            #
            # print('\\\\')
            # print()
            # print('\\vspace{1em}')
            #
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + b_med_png + '} \\\\ Baseline + Median')
            # print('\\end{minipage}')
            # print('\\hfill')
            # print('\\begin{minipage}{0.32\\textwidth}\center')
            # print('\\includegraphics[width=1\\textwidth]{' + ours_png + '} \\\\ ResU-Net (ours)')
            # print('\\end{minipage}')
            # print('\\hfill* \\strut')
            # print('\\\\')
            # print()
            # print('\\vspace{1em}')
            #
            # print('\\begin{center}')
            # print('Sample: ' + dirname + '\\\\')
            # print('\\vspace{0.5em}')
            # print('Filename: \\url{' + entry['gwy_path'].split('/')[-1].replace('\\', '/') + '}')
            # print('\\end{center}')


            print('\\begin{figure}')
            print('\\hfill')
            print('\\subfloat[L$\\rightarrow$R]{\\includegraphics[width=.28\\linewidth]{' + l_png + '}}\\hfill')
            print('\\subfloat[R$\\rightarrow$L]{\\includegraphics[width=.28\\linewidth]{' + r_png + '}}\\hfill \\strut \\par')
            print('\\begin{center}\\begin{tikzpicture}\\begin{axis}[hide axis, scale only axis, height = 0 pt, width = 0 pt, colormap/gray, '
                  'colorbar horizontal, point meta min = ' + '{:.2f}'.format(lr_min / 1e-9) + ', point meta max = ' + '{:.2f}'.format(lr_max / 1e-9) +
                  ', colorbar style = { width = 0.706666\\linewidth, xtick = data}]')
            print('\\addplot[draw = none] coordinates {(0, 0)};')
            print('\\end{axis} \\end{tikzpicture} \\end{center}')
            print('\\hfill')
            print('\\subfloat[Baseline]{\\includegraphics[width=.28\\linewidth]{' + b_png + '}}\\hfill')
            print('\\subfloat[Baseline + Average]{\\includegraphics[width=.28\\linewidth]{' + b_avg_png + '}}\\hfill')
            print('\\subfloat[Baseline + Gauss]{\\includegraphics[width=.28\\linewidth]{' + b_gauss_png + '}}\\hfill \\strut \\par')
            print('\\hfill')
            print('\\subfloat[Baseline + Median]{\\includegraphics[width=.28\\linewidth]{' + b_med_png + '}}\\hfill')
            print('\\subfloat[ResU-Net (ours)]{\\includegraphics[width=.28\\linewidth]{' + ours_png + '}}\\hfill \\strut \\par')
            print('\\begin{center}\\begin{tikzpicture}\\begin{axis}[hide axis, scale only axis, height = 0 pt, width = 0 pt, colormap/gray, '
                  'colorbar horizontal, point meta min = ' + '{:.2f}'.format(img_min / 1e-9) + ', point meta max = ' + '{:.2f}'.format(img_max / 1e-9) +
                  ', colorbar style = { width = 0.706666\\linewidth, xtick = data}]')
            print('\\addplot[draw = none] coordinates {(0, 0)};')
            print('\\end{axis} \\end{tikzpicture} \\end{center}')
            # print('\\caption{Sample: '+ dirname + ' Filename: \\protect\\url{' + entry['gwy_path'].split('/')[-1].replace('\\', '/') + '}}')
            print('\\caption{Sample: '+ dirname + ' Filename: \\protect\\url{' + os.path.basename(entry['gwy_path']) + '}}')
            print('\\end{figure}')



            # print('\\\\')
            # print()
            # print('\\vspace{1em}')
            #
            # print('\\hfill')


            print('\\clearpage')

    print('\\end{document}')

def main(args):
    # print("Checking dirs: ", args.data_path)
    dirs = [dir for dir in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, dir))]

    list_of_entries = []
    for dir in dirs:
        entries_path = os.path.join(args.data_path, dir, 'entries.pkl')
        # print("Loading data from path: ", entries_path)
        data_basename = dir
        with open(entries_path, 'rb') as f:
            entries = pickle.load(f)
        list_of_entries.append(entries)

    list_of_entries = [apply_baseline(entries, threshold=args.threshold, level=args.level) for entries in list_of_entries]

    model = ResUnet(2).cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    list_of_entries = [inference(model, entries, level=args.level) for entries in list_of_entries]

    output_images(list_of_entries)



if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    args = parse_command_line()
    main(args)