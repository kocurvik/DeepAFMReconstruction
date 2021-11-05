import argparse
import itertools
import os

import numpy as np
import cv2
import torch

from eval.registration import register_affine_orb, register_affine_sitk, resample_images
from network.train import load_model
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-r', '--resume', type=int, default=None, help='checkpoint to resume from')
    # parser.add_argument('-nw', '--workers', type=int, default=0, help='workers')
    # parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    # parser.add_argument('--no_preload', action='store_true', default=False)
    # parser.add_argument('-e', '--epochs', type=int, default=250, help='max number of epochs')
    # parser.add_argument('-g', '--gpu', type=str, default='0', help='which gpu to use')
    # parser.add_argument('-de', '--dump_every', type=int, default=0, help='save every n frames during extraction scripts')
    parser.add_argument('path')
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



def get_metric(img_1, img_2, metric='mse', display=None):
    # affine_matrix = register_affine_orb(img_1, img_2)
    # print(affine_matrix)
    transform, metric_value = register_affine_sitk(img_1, img_2, metric=metric, verbose=False)

    img_2_t, img_c = resample_images(img_1, img_2, transform)

    if display is not None:
        cv2.imshow("Orig img 1", img_1)
        cv2.imshow("Orig img 2", img_2)
        cv2.imshow("Transformed img 2", img_2_t)
        cv2.imshow("Composite img", img_c)
        cv2.waitKey(display)

    return metric_value, img_c


def generate_table(img_list_baseline, img_list_nn):
    reg_images = []

    for idx_1, idx_2 in itertools.combinations(np.arange(len(img_list_baseline)), 2):
        print("Images: {} and {}".format(idx_1, idx_2))
        for metric in ['mse', 'correlation', 'mmi']:
            print('\t Metric ', metric)
            metric_value, reg_image_baseline = get_metric(img_list_baseline[idx_1], img_list_baseline[idx_2], metric=metric, display=None)
            print('\t \t Baseline: {}'.format(metric_value))

            metric_value, reg_image_nn = get_metric(img_list_nn[idx_1], img_list_nn[idx_2], metric=metric, display=None)
            print('\t \t NN: {}'.format(metric_value))

            mosaic = np.concatenate([reg_image_baseline, reg_image_nn], axis=1)
            cv2.imshow("vis", mosaic)
            cv2.waitKey(1)
            cv2.imwrite('vis/reg_{}_{}_{}.png'.format(metric, idx_1, idx_2), (255 * mosaic).astype(np.float32))

    return reg_images


def eval_group(model, group_path):
    group_l_names = sorted([name for name in os.listdir(group_path) if 'L.jpg' in name or 'L.png' in name])
    group_r_names = sorted([name for name in os.listdir(group_path) if 'R.jpg' in name or 'R.png' in name])

    baseline_imgs = []
    nn_imgs = []

    for l_name, r_name in zip(group_l_names, group_r_names):
        img_l = normalize(cv2.cvtColor(cv2.imread(os.path.join(group_path, l_name)), cv2.COLOR_BGR2GRAY))
        img_r = normalize(cv2.cvtColor(cv2.imread(os.path.join(group_path, r_name)), cv2.COLOR_BGR2GRAY))

        img_l, img_r = enforce_img_size_for_nn(img_l, img_r)
        input = torch.from_numpy(np.stack([img_l, img_r], axis=0)[np.newaxis, ...])
        nn_img = model(input.cuda()).detach().cpu().numpy()
        nn_imgs.append(np.squeeze(nn_img))

        baseline_img = normalize(baseline_lr_filtering(img_l, img_r))
        baseline_imgs.append(baseline_img)

    generate_table(nn_imgs, baseline_imgs)


def extract_neno_eval_data(model, path):
    filenames = os.listdir(path)
    filenames = [f for f in filenames if '.gwy' in f and 'tip' not in f and 'bad' not in f]

    data_dict = {}

    for filename in filenames:
        prefix = filename.split('_')[0]

        if prefix not in data_dict.keys():
            data_dict[prefix] = []

        gwy_path = os.path.join(path, filename)
        img_l, img_r = load_lr_img_from_gwy(gwy_path)

        input = torch.from_numpy(np.stack([img_l, img_r], axis=0)[np.newaxis, ...]).cuda()
        img_nn = np.squeeze(model(input.cuda()).detach().cpu().numpy())
        img_baseline = normalize(baseline_lr_filtering(img_l, img_r))

        # cv2.imshow('Img L', img_l)
        # cv2.imshow('Img R', img_r)
        # cv2.imshow('Img baseline', img_baseline)
        # cv2.imshow('Img NN', img_nn)
        # cv2.waitKey(0)

        entry_dict = {'filename': filename, 'gwy_path': gwy_path, 'img_l': img_l, 'img_r': img_r, 'img_nn': img_nn, 'img_baseline': img_baseline}

        data_dict[prefix].append(entry_dict)

    return data_dict


def eval_same_sample(dict_list):
    for idx_1, idx_2 in itertools.combinations(np.arange(len(dict_list)), 2):
        print("Images: {} and {}".format(dict_list[idx_1]['filename'], dict_list[idx_2]['filename']))
        for metric in ['mse', 'correlation', 'mmi']:
            print('\t Metric ', metric)
            metric_value, reg_image_baseline = get_metric(dict_list[idx_1]['img_baseline'], dict_list[idx_2]['img_baseline'], metric=metric, display=None)
            print('\t \t Baseline: {}'.format(metric_value))

            metric_value, reg_image_nn = get_metric(dict_list[idx_1]['img_nn'], dict_list[idx_2]['img_nn'], metric=metric, display=1)
            print('\t \t NN: {}'.format(metric_value))

            # mosaic = np.concatenate([reg_image_baseline, reg_image_nn], axis=1)
            # cv2.imshow("vis", mosaic)
            # cv2.waitKey(1)
            # cv2.imwrite('vis/reg_{}_{}_{}.png'.format(metric, idx_1, idx_2), (255 * mosaic).astype(np.float32))


def eval_neno(model, path):
    data_dict = extract_neno_eval_data(model, path)

    for prefix in data_dict.keys():
        print(10 * '*' + ' Evaluating sample ' + prefix + ' ' + 10 * '*')
        eval_same_sample(data_dict[prefix])


if __name__ == '__main__':
    args = parse_command_line()

    model = load_model(args)
    model.eval()

    # eval_group(model, args.path)
    eval_neno(model, args.path)