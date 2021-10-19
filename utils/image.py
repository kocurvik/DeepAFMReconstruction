import os.path

import cv2
import numpy as np
import gwyfile
import torch


def normalize(img):
    img = img.astype(np.float32)
    if img.max() == img.min():
        return img
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def remove_offset_lr(img_l, img_r, max_offset=64):
    mses = np.zeros(max_offset)
    max_width = img_l.shape[1]
    for offset in range(max_offset):
        diff = img_l[:, offset:max_width - max_offset + offset] - img_r[:, : max_width - max_offset]
        mses[offset] = np.mean(diff ** 2)

    best_offset = np.argmin(mses)
    return img_l[:, best_offset:], img_r[:, :max_width - best_offset]


def enforce_img_size_for_nn(img_1, img_2, dim=8):
    min_height = (min(img_1.shape[0], img_2.shape[0]) // dim) * dim
    min_width = (min(img_1.shape[1], img_2.shape[1]) // dim) * dim

    img_1 = img_1[:min_height, :min_width]
    img_2 = img_2[:min_height, :min_width]
    return img_1, img_2


def load_lr_img_from_gwy(gwy_path, remove_offset=True, normalize_range=True, enforce_nn=True):
    obj = gwyfile.load(gwy_path)
    channels = gwyfile.util.get_datafields(obj)

    basename = os.path.basename(gwy_path)

    img_r = channels['Topo [<]'].data
    img_l = channels['Topo [>]'].data

    if 'l-r' in basename or 'r-l' in basename:
        img_r = np.rot90(img_r)
        img_l = np.rot90(img_l)

    if normalize_range:
        img_l = normalize(img_l)
        img_r = normalize(img_r)

    if remove_offset:
        img_l, img_r = remove_offset_lr(img_l, img_r)

    if enforce_nn:
        img_l, img_r = enforce_img_size_for_nn(img_l, img_r)

    return img_l, img_r


# if __name__ == '__main__':
#     # img_l_td, img_r_td = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/Neno_t-d_0deg_60deg-scanner_210330_134900.gwy')
#     # img_l_lr, img_r_lr = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/Neno_l-r_0deg_90deg-scanner_210330_132513.gwy')
#     img_l_td, img_r_td = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGZ3_t-d_0deg_45deg-scanner_210326_151151.gwy')
#     img_l_lr, img_r_lr = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGZ3_l-r_0deg_45deg-scanner_210326_153256.gwy')
#     img_l_bu, img_r_bu = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/TGQ1_b-u_0deg_0deg-scanner_210326_105247.gwy')
#     img_l_rl, img_r_rl = load_lr_img_from_gwy('D:/Research/data/GEFSEM/2021-04-07 - Dataset/Neno_r-l_45deg_0deg-scanner_210330_143917.gwy')
#
#     cv2.imshow("img_l_td", img_l_td)
#     cv2.imshow("img_r_td", img_r_td)
#     cv2.imshow("img_l_lr", img_l_lr)
#     cv2.imshow("img_r_lr", img_r_lr)
#     cv2.imshow("img_l_bu", img_l_bu)
#     cv2.imshow("img_r_bu", img_r_bu)
#     cv2.imshow("img_l_rl", img_l_rl)
#     cv2.imshow("img_r_rl", img_r_rl)
#     cv2.waitKey(0)