import os.path
import pickle

import cv2
import numpy as np
import gwyfile
from matplotlib import pyplot as plt
from scipy import ndimage


def normalize(img):
    # Normalize image to range [0, 1]
    img = img.astype(np.float32)
    if img.max() == img.min():
        return img
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def subtract_mean_plane(img, return_plane=False):
    # Subtract mean plane from single image
    x, y = np.mgrid[:img.shape[0], :img.shape[1]]

    X = np.column_stack([x.ravel(), y.ravel(), np.ones(img.shape[0] * img.shape[1])])
    H = img.ravel()

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), H)
    plane = np.reshape(np.dot(X, theta), (img.shape[0], img.shape[1]))

    if return_plane:
        return img - plane, plane
    return img - plane


def subtract_mean_plane_both(img_l, img_r, return_plane=False):
    # Subtract mean plane from two images. The mean plane is calculated for both images simultaneously
    x, y = np.mgrid[:img_l.shape[0], :img_l.shape[1]]
    X = np.column_stack([x.ravel(), y.ravel(), np.ones(img_l.shape[0] * img_l.shape[1])])
    X = np.concatenate([X, X], axis=0)
    H = np.concatenate([img_l.ravel(), img_r.ravel()], axis=0)

    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), H)
    plane = np.reshape(np.dot(X, theta), (2 * img_l.shape[0], img_r.shape[1]))

    plane = plane[:img_l.shape[0], :]

    if return_plane:
        return img_l - plane, img_r - plane, plane
    return img_l - plane, img_r - plane


def normalize_joint(imgs):
    # Normalize a list of images so that the min value is 0 and max value is 1
    joint = np.stack(imgs, axis=0)
    max = np.max(joint)
    min = np.min(joint)

    if max == min:
        return np.zeros_like(imgs)

    n_imgs = []
    for img in imgs:
        n_imgs.append((img - min) / (max - min))

    return n_imgs


def denormalize(img, orig_imgs):
    # Reverse normalization of img based on min and max values of orig_imgs
    joint_orig = np.stack(orig_imgs, axis=0)
    max = np.max(joint_orig)
    min = np.min(joint_orig)

    return (img * (max - min)) + min


def remove_offset_lr(img_l, img_r, max_offset=64):
    # Align two images img_l and img_r so that they overlap the best w.r.t. mutual MSE, then crop the images so only
    # the overlapped area remains
    mses = np.zeros(max_offset)
    max_width = img_l.shape[1]
    for offset in range(max_offset):
        diff = img_l[:, offset:max_width - max_offset + offset] - img_r[:, : max_width - max_offset]
        mses[offset] = np.mean(diff ** 2)

    best_offset = np.argmin(mses)

    return img_l[:, best_offset:], img_r[:, :max_width - best_offset]


def enforce_img_size_for_nn(img_1, img_2, dim=8):
    # Make sure that the dimensions of the image can be passed to the NN
    # E.g. dims of img_1 and img_2 have to be divisible by dim)
    min_height = (min(img_1.shape[0], img_2.shape[0]) // dim) * dim
    min_width = (min(img_1.shape[1], img_2.shape[1]) // dim) * dim

    img_1 = img_1[:min_height, :min_width]
    img_2 = img_2[:min_height, :min_width]
    return img_1, img_2


def load_lr_img_from_gwy(gwy_path, remove_offset=True, normalize_range=True, enforce_nn=True):
    # Load the left and right topography images from a .gwy file. Additional args enable the use of more postprocessing
    obj = gwyfile.load(gwy_path)
    channels = gwyfile.util.get_datafields(obj)

    img_r = channels['Topo [<]'].data
    img_l = channels['Topo [>]'].data

    scan_direction = obj['/0/meta']['scan.dir']

    # basename = os.path.basename(gwy_path)
    # if 'l-r' in basename or 'r-l' in basename:
    if 'left-right' == scan_direction or 'right-left' == scan_direction:
        img_r = np.rot90(img_r)
        img_l = np.rot90(img_l)

    if normalize_range:
        img_l, img_r = normalize(np.stack([img_l, img_r], axis=0))

    if remove_offset:
        img_l, img_r = remove_offset_lr(img_l, img_r)

    if enforce_nn:
        img_l, img_r = enforce_img_size_for_nn(img_l, img_r)

    return img_l, img_r


def rotate(image, deg):
    # Rotate image by deg degrees
    return ndimage.rotate(image, deg, reshape=True)


def load_tips_from_pkl(pkl_path):
    # Load tip topographies from a pkl file
    with open(pkl_path, 'rb') as f:
        tips = pickle.load(f)
        tips = tips[list(tips.keys())[0]]
    return tips

if __name__ == '__main__':
    # Some code to visually check the implemented methods

    # img_l, img_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalData/Tescan sample/4x4_l-r_+90deg_210908_145519.gwy')
    img_l, img_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalData/Neno/Neno_r-l_45deg_0deg-scanner_210330_143917.gwy')
    img_l, img_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalNew/loga/loga_0deg_220126_155543.gwy')
    # img_l, img_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalSlow/kremik_5x5_t-d_0deg_slow.gwy')
    # img_l, img_r = load_lr_img_from_gwy('D:/Research/data/GEFSEM/EvalSlow/kremik_5x5_t-d_0deg.gwy')

    for i in range(0, 512, 16):
        print(i)
        disp_l = np.copy(img_l)
        disp_r = np.copy(img_r)

        disp_l[i, :] = 0
        disp_r[i, :] = 0

        plt.clf()
        plt.plot(img_l[i, :], c='r')
        plt.plot(img_r[i, :], c='b')
        # plt.plot(img_baseline[i, :], c='k')
        plt.pause(0.1)

        cv2.imshow("img_l_td", disp_l)
        cv2.imshow("img_r_td", disp_r)
        cv2.imshow("img_r_td_plane", normalize(subtract_mean_plane(img_l)))
        cv2.waitKey(0)



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