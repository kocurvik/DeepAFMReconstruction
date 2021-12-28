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


def get_metric(img_1, img_2, metric='mse', p1=None, p2=None, display=None):
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
        cv2.waitKey(display)

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

    results['mse'] = np.ones([len(entries), len(entries)])
    results['rmse'] = np.ones([len(entries), len(entries)])
    results['correlation'] = np.ones([len(entries), len(entries)])
    results['rcorrelation'] = np.ones([len(entries), len(entries)])

    # results['correlation'] = np.ones([len(entries), len(entries)])

    for idx_1, idx_2 in itertools.combinations(np.arange(len(entries)), 2):
        print("Images: {} and {}".format(entries[idx_1]['filename'], entries[idx_2]['filename']))
        for metric in ['mse', 'correlation']:
        # for metric in ['mse']:
            metric_value, reg_image_nn = get_metric(entries[idx_1]['img_out'], entries[idx_2]['img_out'],
                                                    p1=entries[idx_1]['registration_points'],
                                                    p2=entries[idx_2]['registration_points'],
                                                    metric=metric, display=1)
            print('\t \t {}: {}, sqrt: {}'.format(metric, metric_value, np.sqrt(metric_value)))
            results[metric][idx_1, idx_2] = metric_value
            results[metric][idx_2, idx_1] = metric_value
            results['r{}'.format(metric)][idx_1, idx_2] = np.sqrt(metric_value)
            results['r{}'.format(metric)][idx_2, idx_1] = np.sqrt(metric_value)

    results['mse'] = results['mse'].tolist()
    results['rmse'] = results['rmse'].tolist()
    results['correlation'] = results['correlation'].tolist()
    results['rcorrelation'] = results['rcorrelation'].tolist()

    return results


def inference(model, entries):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        # img_nn = get_multi_input(model, img_l_normalized, img_r_normalized)
        nn_input = torch.from_numpy(np.stack([img_l_normalized, img_r_normalized], axis=0)[None, ...]).float().cuda()
        img_nn = model(nn_input).detach().cpu().numpy()[0, 0, ...]
        entry['img_out_normalized'] = img_nn
        entry['img_out'] = denormalize(img_nn, [img_l, img_r])
    return entries


def apply_baseline(entries):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        img_l_normalized, img_r_normalized = normalize_joint([img_l, img_r])
        img_baseline = normalize(baseline_lr_filtering(img_l_normalized, img_r_normalized))
        entry['img_out'] = denormalize(img_baseline, [img_l, img_r])
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


def output_images(list_of_entries, model_basename):
    pdf_imgs = []
    for entries in list_of_entries:
        for entry in entries:
            images = [Image.fromarray((255 * normalize(entry[x])).astype(np.uint8)) for x in ['img_l', 'img_r', 'img_out']]
            img_new = compose_images(images)

            pdf_imgs.append(img_new)

    pdf_imgs[0].save("vis/images_{}.pdf".format(model_basename), "PDF", resolution=100.0, save_all=True, append_images=pdf_imgs[1:])


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

    if args.model_path == 'baseline':
        list_of_entries = [apply_baseline(entries) for entries in list_of_entries]
        model_basename = 'baseline'
    else:
        model = ResUnet(2).cuda()
        print("Resuming from: ", args.model_path)
        model_basename = os.path.basename(args.model_path).split('.')[0]
        model.load_state_dict(torch.load(args.model_path))
        model.eval()
        list_of_entries = [inference(model, entries) for entries in list_of_entries]

    if args.images:
        output_images(list_of_entries, model_basename)

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