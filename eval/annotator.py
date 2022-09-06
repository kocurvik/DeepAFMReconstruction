import argparse
import os
import pickle

import cv2
import numpy as np

from utils.image import load_lr_img_from_gwy, normalize, enforce_img_size_for_nn


def parse_command_line():
    """ Parser used for training and inference returns args. Sets up GPUs."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual_offset', action='store_true', default=False, help='Whether the offset needs to be set manually')
    parser.add_argument('--mask', action='store_true', default=False, help='Whether the load the masks')
    parser.add_argument('data_path', help='Path to the dataset folder containing multiple gwy files of the same sample')
    args = parser.parse_args()
    return args


def annotate_entries(entries):
    # Annotate the images by selecting the same keypoint in all images
    for i, entry in enumerate(entries):

        print("Entry filename: ", entry['filename'])

        if i > 0:
            cv2.imshow("Chosen points!", example_img)

        clicked = []
        def mouse_callback(event, x, y, flags, params):
            if event == 1:
                clicked.append([x, y])
                print("Clicked x: {}, y: {}".format(x, y))

        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('image', mouse_callback)
        cv2.imshow('image', normalize(entry['img_l']))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        entry['registration_points'] = clicked

        if i == 0:
            example_img = cv2.cvtColor(normalize(entry['img_l']), cv2.COLOR_GRAY2BGR)
            for pt in clicked:
                example_img = cv2.circle(example_img, (pt[0], pt[1]), 3, (0, 0, 255), thickness=2)

    return entries


def set_manual_offset(img_l, img_r, mask=None):
    # Set manual offset for img_l and img_r when alignment via simple MSE fails.
    # Controlled by key presses: t and v control contrast, k and s control the offset,
    # c continues to next image and saves offset
    offset = 0
    gamma = 1

    max_width = img_r.shape[1]

    cv2.namedWindow('composite', cv2.WINDOW_NORMAL)

    while True:
        img_l_off, img_r_off = img_l[:, offset:], img_r[:, :max_width - offset]

        print('MSE: ', np.sqrt(np.mean((img_l_off - img_r_off) ** 2)))

        composite = np.stack([normalize(img_l_off) ** gamma, np.zeros_like(img_r_off), normalize(img_r_off) ** gamma], axis=-1)
        cv2.imshow('img_l', normalize(img_l_off))
        cv2.imshow('img_r', normalize(img_r_off))
        cv2.imshow('composite', composite)

        key = cv2.waitKey(0)
        if key == ord('k'):
            offset = min(max_width - 1, offset + 1)
        if key == ord('s'):
            offset = max(0, offset - 1)
        if key == ord('t'):
            gamma = min(3, gamma + 0.1)
        if key == ord('v'):
            gamma = max(0.1, gamma - 0.1)
        if key == ord('c'):
            if mask is None:
                return enforce_img_size_for_nn(img_l_off, img_r_off)
            return enforce_img_size_for_nn(img_l_off, img_r_off, mask[:, offset:])


def extract_eval_data(path, manual_offset=False, load_mask=False):
    # Loads gwy data from a folder
    filenames = os.listdir(path)
    filenames = [f for f in filenames if '.gwy' in f and 'tip' not in f and 'bad' not in f]

    entries = []

    for filename in filenames:
        print("Loading:", filename)
        gwy_path = os.path.join(path, filename)
        if load_mask:
            if manual_offset:
                img_l, img_r, mask = load_lr_img_from_gwy(gwy_path, remove_offset=False, normalize_range=False, include_mask=True)
                img_l, img_r, mask = set_manual_offset(img_l, img_r, mask=mask)
            else:
                img_l, img_r, mask = load_lr_img_from_gwy(gwy_path, normalize_range=False, include_mask=True)

            entry_dict = {'filename': filename, 'gwy_path': gwy_path, 'img_l': img_l, 'img_r': img_r, 'mask': mask}
        else:
            if manual_offset:
                img_l, img_r = load_lr_img_from_gwy(gwy_path, remove_offset=False, normalize_range=False)
                img_l, img_r = set_manual_offset(img_l, img_r)
            else:
                img_l, img_r = load_lr_img_from_gwy(gwy_path, normalize_range=False)

            entry_dict = {'filename': filename, 'gwy_path': gwy_path, 'img_l': img_l, 'img_r': img_r}

        entries.append(entry_dict)
    return entries


def save_entries(entries, path):
    pkl_path = os.path.join(path, 'entries.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(entries, f)


if __name__ == '__main__':
    args = parse_command_line()

    entries = extract_eval_data(args.data_path, manual_offset=args.manual_offset, load_mask=args.mask)
    annotate_entries(entries)
    save_entries(entries, args.data_path)