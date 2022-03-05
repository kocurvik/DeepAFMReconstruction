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
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy, normalize_joint, denormalize, \
    subtract_mean_plane, subtract_mean_plane_both

DIRNAME_DICT = {'d008': 'Wafers', 'D010_Bunky': 'Cells', 'INCHAR (MFM sample)': 'Permalloy', 'Kremik': 'Silicon',
                'loga': 'Logos', 'Neno': 'Neno', 'Tescan sample': 'Patterns', 'TGQ1': 'TGQ1', 'TGZ3': 'TGZ3'}

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


def get_metric(entry_1, entry_2, metric='mse', p1=None, p2=None, display=None):
    img_1 = entry_1['img_out']
    img_2 = entry_2['img_out']
    p1 = entry_1['registration_points']
    p2 = entry_2['registration_points']

    if metric == 'correlation':
        transform, metric_value = register_rigid_sitk(normalize(img_1), normalize(img_2), metric=metric, p1=p1, p2=p2, verbose=False)
        metric_value = -metric_value
    else:
        transform, metric_value = register_rigid_sitk(img_1, img_2, metric=metric, p1=p1, p2=p2, verbose=False)


    img_2_t, img_c = resample_images(normalize(img_1), normalize(img_2), transform)

    if display is not None:
        cv2.imshow("Orig img 1", normalize(img_1))
        cv2.imshow("Orig img 2", normalize(img_2))
        cv2.imshow("Transformed img 2", normalize(img_2_t))
        cv2.imshow("Composite img", normalize(img_c))

        key = cv2.waitKey(display)
        if key == ord('s'):
            cv2.imwrite("figs/eval/orig_1.png", (255 * normalize(img_1)).astype(np.uint8))
            cv2.imwrite("figs/eval/lr_1.png", (255 * normalize(entry_1['img_l'])).astype(np.uint8))
            cv2.imwrite("figs/eval/rl_1.png", (255 * normalize(entry_1['img_r'])).astype(np.uint8))

            cv2.imwrite("figs/eval/orig_2.png", (255 * normalize(img_2)).astype(np.uint8))
            cv2.imwrite("figs/eval/lr_2.png", (255 * normalize(entry_2['img_l'])).astype(np.uint8))
            cv2.imwrite("figs/eval/rl_2.png", (255 * normalize(entry_2['img_r'])).astype(np.uint8))
            cv2.imwrite("figs/eval/transformed_2.png", (255 * normalize(img_2_t)).astype(np.uint8))

            cv2.imwrite("figs/eval/composite.png", (255 * normalize(img_c)).astype(np.uint8))

    return metric_value, img_c


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


def eval_same_sample(entries):
    results = {'filenames': {i: entry['filename'] for i, entry in enumerate(entries)}}

    # results['correlation'] = np.ones([len(entries), len(entries)])

    for idx_1, idx_2 in itertools.combinations(np.arange(len(entries)), 2):
        print("Images: {} and {}".format(entries[idx_1]['filename'], entries[idx_2]['filename']))
        # for metric in ['mse', 'correlation']:

        cv2.imshow('img_1_out', normalize(entries[idx_1]['img_out']))
        cv2.imshow('img_2_out', normalize(entries[idx_2]['img_out']))
        key = cv2.waitKey(0)
        if key != ord('s'):
            continue

        metric_value, reg_image_nn = get_metric(entries[idx_1], entries[idx_2], metric='mse', display=0)

    return results


def inference(model, entries, level=False):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        # if 'slow' in entry['filename']:
        #     entry['img_out'] = img_r.astype(np.float32)
        #     continue

        if level:
            img_l, img_r = subtract_mean_plane_both(img_l, img_r)

        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        # img_nn = get_multi_input(model, img_l_normalized, img_r_normalized)
        nn_input = torch.from_numpy(np.stack([img_l_normalized, img_r_normalized], axis=0)[None, ...]).float().cuda()
        img_nn = model(nn_input).detach().cpu().numpy()[0, 0, ...]
        if level:
            entry['img_out'] = subtract_mean_plane(denormalize(img_nn, [img_l, img_r]))
            entry['img_out_normalized'] = normalize(subtract_mean_plane(img_nn))
        else:
            entry['img_out_normalized'] = img_nn
            entry['img_out'] = denormalize(img_nn, [img_l, img_r])
    return entries


def apply_baseline(entries, gauss=False, average=False, median=False, threshold=0.1, level=False):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        # if 'slow' in entry['filename']:
        #     entry['img_out'] = img_r.astype(np.float32)
        #     continue

        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        img_baseline = normalize(baseline_lr_filtering(img_l_normalized, img_r_normalized, threshold=threshold))
        if gauss:
            img_baseline = cv2.GaussianBlur(img_baseline, (5, 5), 0)
        if average:
            img_baseline = cv2.blur(img_baseline, (5, 5))
        if median:
            img_baseline = cv2.medianBlur(img_baseline, 5)

        if level:
            entry['img_out_baseline'] = subtract_mean_plane(denormalize(img_baseline, [img_l, img_r]))
        else:
            entry['img_out_baseline'] = denormalize(img_baseline, [img_l, img_r])
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
    pdf_imgs = []

    print('\\begin{tabular}{m{0.02\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth} m{0.24\\textwidth}}')

    for entries in list_of_entries:
        for entry in entries:
            dirname = os.path.basename(os.path.dirname(entry['gwy_path']))

            baseline_img, ours_img = normalize_joint([entry['img_out_baseline'], entry['img_out']])
            img_l, img_r = normalize_joint([entry['img_l'], entry['img_r']])

            cv2.imshow('l', normalize(img_l))
            cv2.imshow('r', normalize(img_r))
            cv2.imshow('out_ours', ours_img)
            cv2.imshow('out_baseline', baseline_img)

            print('MSE: ', np.sqrt(np.mean((entry['img_out'] - entry['img_out_baseline'])**2)))

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


def output_aligned_images(list_of_entries, model_basename):
    pdf_imgs = []
    for entries in list_of_entries:
        entry_first = entries[0]
        images = [Image.fromarray((255 * normalize(entry_first[x])).astype(np.uint8)) for x in ['img_l', 'img_r', 'img_out']]
        pdf_imgs.append(compose_images(images))

        for idx in range(1, len(entries)):
            images = []
            entry = entries[idx]
            for img_key in ['img_l', 'img_r', 'img_out']:
                orig_img = entry_first[img_key]
                t_img = entry[img_key]
                transform, metric_value = register_rigid_sitk(orig_img, t_img, metric='mse', p1=entry_first['registration_points'], p2=entry['registration_points'], verbose=False)
                img_t, _ = resample_images(normalize(orig_img), normalize(t_img), transform)
                images.append(Image.fromarray((255 * normalize(img_t)).astype(np.uint8)))

            pdf_imgs.append(compose_images(images))

    pdf_imgs[0].save("vis/aligned_{}.pdf".format(model_basename), "PDF", resolution=100.0, save_all=True, append_images=pdf_imgs[1:])


def main(args):
    print("Checking dirs: ", args.data_path)
    dirs = [dir for dir in os.listdir(args.data_path) if os.path.isdir(os.path.join(args.data_path, dir))]

    list_of_entries = []
    for dir in dirs:
        entries_path = os.path.join(args.data_path, dir, 'entries.pkl')
        print("Loading data from path: ", entries_path)
        data_basename = dir
        with open(entries_path, 'rb') as f:
            entries = pickle.load(f)
        list_of_entries.append(entries)

    # if args.model_path == 'baseline':
    list_of_entries = [apply_baseline(entries, args.gauss, args.average, args.median, threshold=args.threshold, level=args.level) for entries in list_of_entries]
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
    list_of_entries = [inference(model, entries, level=args.level) for entries in list_of_entries]

    if args.level:
        model_basename += '_level'

    print("Baseline model basename: " + baseline_model_basename)
    print("Model basename: " + model_basename)

    if args.images:
        output_images(list_of_entries)

    if args.aligned_images:
        output_aligned_images(list_of_entries, model_basename)

    if args.eval:
        list_of_results = []
        for entries, dir in zip(list_of_entries, dirs):
            results = eval_same_sample(entries)
            results['dir'] = dir
            list_of_results.append(results)

        with open(os.path.join(args.data_path, '{}_results.json'.format(model_basename)), 'w') as f:
            json.dump(list_of_results, f, indent=4)


if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    args = parse_command_line()
    main(args)