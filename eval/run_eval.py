import argparse
import itertools
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
from utils.image import normalize, enforce_img_size_for_nn, load_lr_img_from_gwy


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
    transform, metric_value = register_rigid_sitk(img_1, img_2, metric=metric, p1=p1, p2=p2, verbose=False)

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
            output_tile = model(torch.from_numpy(input).cuda()).detach().cpu().numpy()

            output[i*tile_size: i*tile_size + 128, j*tile_size: j*tile_size + 128] += output_tile[0, 0]
            output_counts[i*tile_size: i*tile_size + 128, j*tile_size: j*tile_size + 128] += 1

    return output/output_counts


def eval_group(model, group_path):
    group_l_names = sorted([name for name in os.listdir(group_path) if 'L.jpg' in name or 'L.png' in name])
    group_r_names = sorted([name for name in os.listdir(group_path) if 'R.jpg' in name or 'R.png' in name])

    baseline_imgs = []
    nn_imgs = []

    for l_name, r_name in zip(group_l_names, group_r_names):
        img_l = normalize(cv2.cvtColor(cv2.imread(os.path.join(group_path, l_name)), cv2.COLOR_BGR2GRAY))
        img_r = normalize(cv2.cvtColor(cv2.imread(os.path.join(group_path, r_name)), cv2.COLOR_BGR2GRAY))

        img_l, img_r = enforce_img_size_for_nn(img_l, img_r)
        # input = torch.from_numpy(np.stack([img_l, img_r], axis=0)[np.newaxis, ...])
        # nn_img = model(input.cuda()).detach().cpu().numpy()
        nn_img = get_multi_input(model, img_l, img_r)

        nn_imgs.append(np.squeeze(nn_img))

        baseline_img = normalize(baseline_lr_filtering(img_l, img_r))
        baseline_imgs.append(baseline_img)

    generate_table(nn_imgs, baseline_imgs)


def eval_same_sample(entries):
    ratios = []
    for idx_1, idx_2 in itertools.combinations(np.arange(len(entries)), 2):
        print("Images: {} and {}".format(entries[idx_1]['filename'], entries[idx_2]['filename']))
        # for metric in ['mse', 'correlation', 'mmi']:
        for metric in ['mse']:
        # for metric in ['correlation']:
            print('\t Metric ', metric)

            baseline_metric_value, reg_image_baseline = get_metric(entries[idx_1]['img_baseline'], entries[idx_2]['img_baseline'],
                                                          p1=entries[idx_1]['registration_points'], p2=entries[idx_2]['registration_points'],
                                                          metric=metric, display=None)
            print('\t \t Baseline: {}'.format(baseline_metric_value))

            nn_metric_value, reg_image_nn = get_metric(entries[idx_1]['img_nn'], entries[idx_2]['img_nn'],
                                                    p1=entries[idx_1]['registration_points'], p2=entries[idx_2]['registration_points'],
                                                    metric=metric, display=1)
            print('\t \t NN: {}'.format(nn_metric_value))

            ratios.append(nn_metric_value / baseline_metric_value)

            # mosaic = np.concatenate([reg_image_baseline, reg_image_nn], axis=1)
            # cv2.imshow("vis", mosaic)
            # cv2.waitKey(1)
            # cv2.imwrite('vis/reg_{}_{}_{}.png'.format(metric, idx_1, idx_2), (255 * mosaic).astype(np.float32))

    print("Ratios: ", ratios)
    print("Ratios mean: ", np.mean(ratios))
    print("Ratios median: ", np.median(ratios))
    print("Ratios geometric mean: ", np.exp(np.log(ratios).mean()))


def inference(model, entries):
    for entry in entries:
        img_l = entry['img_l']
        img_r = entry['img_r']
        # input = torch.from_numpy(np.stack([img_l, img_r], axis=0)[np.newaxis, ...]).cuda()
        # img_nn = np.squeeze(model(input.cuda()).detach().cpu().numpy())
        img_nn = get_multi_input(model, img_l, img_r)
        img_baseline = normalize(baseline_lr_filtering(img_l, img_r))
        entry['img_nn'] = img_nn
        entry['img_baseline'] = img_baseline
    return entries


def output_images(entries, data_basename, model_basename):
    pdf_imgs = []
    for entry in entries:
        images = [Image.fromarray((255 * entry[x] / entry[x].max()).astype(np.uint8)) for x in ['img_l', 'img_r', 'img_baseline', 'img_nn']]
        img_new = compose_images(images)

        pdf_imgs.append(img_new)

    pdf_imgs[0].save("vis/images_{}_{}.pdf".format(data_basename, model_basename), "PDF", resolution=100.0, save_all=True, append_images=pdf_imgs[1:])


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


def output_aligned_images(entries, data_basename, model_basename):
    entry_first = entries[0]
    images = [Image.fromarray((255 * entry_first[x] / entry_first[x].max()).astype(np.uint8)) for x in
              ['img_l', 'img_r', 'img_baseline', 'img_nn']]
    pdf_imgs = [compose_images(images)]

    for idx in range(1, len(entries)):
        images = []
        entry = entries[idx]
        for img_key in ['img_l', 'img_r', 'img_baseline', 'img_nn']:
            orig_img = entry_first[img_key]
            t_img = entry[img_key]
            transform, metric_value = register_rigid_sitk(orig_img, t_img, metric='mse', p1=entry_first['registration_points'], p2=entry['registration_points'], verbose=False)
            img_t, _ = resample_images(orig_img, t_img, transform)
            images.append(Image.fromarray((255 * img_t / img_t.max()).astype(np.uint8)))

        pdf_imgs.append(compose_images(images))

    pdf_imgs[0].save("vis/aligned_{}_{}.pdf".format(data_basename, model_basename), "PDF", resolution=100.0, save_all=True, append_images=pdf_imgs[1:])


def main(args):
    model = ResUnet(2).cuda()
    print("Resuming from: ", args.model_path)
    model_basename = os.path.basename(args.model_path).split('.')[0]
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print("Loading data from path: ", args.data_path)
    data_basename = os.path.basename(os.path.dirname(args.data_path))
    with open(args.data_path, 'rb') as f:
        entries = pickle.load(f)

    entries = inference(model, entries)
    if args.images:
        output_images(entries, data_basename, model_basename)

    if args.aligned_images:
        output_aligned_images(entries, data_basename, model_basename)

    if args.eval:
        eval_same_sample(entries)


if __name__ == '__main__':
    # Example usage python eval/run_eval.py -i checkpoints/4e0b.pth "D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1/entries.pkl"
    args = parse_command_line()
    main(args)