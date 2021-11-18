import math
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import binary_dilation

from synth.generator import generate_grid_structure
from utils.image import load_tips_from_pkl, normalize


def dilate(image, kernel):
    """
    Dilates image by kernel
    :param image:
    :param kernel:
    :return:
    """
    image_height = np.shape(image)[0]
    image_width = np.shape(image)[1]
    kernel_height = np.shape(kernel)[0]
    kernel_width = np.shape(kernel)[1]
    kernel_half_height = kernel_height // 2
    kernel_half_width = kernel_width // 2

    mask_height = image_height + kernel_height
    mask_width = image_width + kernel_width
    mask = np.zeros((mask_height, mask_width))
    mask[kernel_half_height: mask_height - (kernel_half_height) - (kernel_height % 2),
         kernel_half_width: mask_width - (kernel_half_width) - (kernel_width % 2)] = image
    ret_image = np.zeros((image_height, image_width))
    kernal_max_ind = np.unravel_index(np.argmax(kernel, axis=None), kernel.shape)

    for i in range(kernel_half_height, image_height + kernel_half_height):
        for j in range(kernel_half_width, image_width + kernel_half_width):
            image_window = mask[i - kernel_half_height: i + (kernel_half_height) + (kernel_height % 2),
                                j - kernel_half_width: j + (kernel_half_width) + (kernel_width % 2)]
            profile = image_window + kernel
            indices = np.unravel_index(np.argmax(profile, axis=None), profile.shape)
            ret_image[i - kernel_half_height, j - kernel_half_width] = image_window[indices] - (kernel[kernal_max_ind] - kernel[indices])

    return ret_image


def fast_dilate(image, kernel):
    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    kernel_half_height = kernel_height // 2
    kernel_half_width = kernel_width // 2
    kernel_max = np.max(kernel)

    mask_height = image_height + kernel_height
    mask_width = image_width + kernel_width
    mask = np.zeros((mask_height, mask_width))
    mask[kernel_half_height: mask_height - (kernel_half_height) - (kernel_height % 2),
         kernel_half_width: mask_width - (kernel_half_width) - (kernel_width % 2)] = image

    image_x_changes = np.zeros_like(mask)
    image_y_changes = np.zeros_like(mask)

    image_x_changes[kernel_half_height: image_height + kernel_half_height,
                    kernel_half_width + 1: image_width + kernel_half_width] = 1 * (image[:, 1:] == image[:, :-1])

    image_y_changes[kernel_half_height + 1: image_height + kernel_half_height,
                    kernel_half_width: image_width + kernel_half_width] = 1 * (image[1:, :] == image[:-1, :])

    # image_x_changes[kernel_half_width: image_width + kernel_half_width, 0] = np.where(image[:, 0] == 0, 1, 0)
    # image_x_changes[kernel_half_width: image_width + kernel_half_width, image_width + kernel_half_width] = np.where(image[:, -1] == 0, 1, 0)
    for i in range(1, image_x_changes.shape[1]):
        image_x_changes[:, i] = (image_x_changes[:, i - 1] + 1) * image_x_changes[:, i]

    # image_y_changes[0, kernel_half_height: image_height + kernel_half_height] = np.where(image[0, :] == 0, 1, 0)
    # image_y_changes[image_height + kernel_half_height, kernel_half_height: image_height + kernel_half_height] = np.where(image[-1, :] == 0, 1, 0)
    for i in range(1, image_y_changes.shape[0]):
        image_y_changes[i, :] = (image_y_changes[i - 1, :] + 1) * image_y_changes[i, :]

    # get mask of positions where dilation is not necessar
    ignore_mask = np.logical_and(image_x_changes >= (kernel_width * 2 + 1), image_y_changes >= (kernel_height * 2 + 1))
    ret_image = np.zeros_like(image)

    for i in range(kernel_half_height, image_height + kernel_half_height):
        for j in range(kernel_half_width, image_width + kernel_half_width):
            if ignore_mask[i + kernel_half_height, j + kernel_half_width]:
                ret_image[i - kernel_half_height, j - kernel_half_width] = mask[i, j]
            else:
                image_window = mask[i - kernel_half_height: i + (kernel_half_height) + (kernel_height % 2),
                                    j - kernel_half_width: j + (kernel_half_width) + (kernel_width % 2)]
                profile = image_window + kernel
                indices = np.unravel_index(np.argmax(profile, axis=None), profile.shape)
                ret_image[i - kernel_half_height, j - kernel_half_width] = image_window[indices] - (kernel_max - kernel[indices])
    return ret_image


def faster_dilate(image, kernel):
    image_height = image.shape[0]
    image_width = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]
    kernel_half_height = kernel_height // 2
    kernel_half_width = kernel_width // 2
    kernel_max = np.max(kernel)

    ret_image = np.zeros_like(image)
    mask_height = image_height + kernel_height
    mask_width = image_width + kernel_width
    mask = np.zeros((mask_height, mask_width))
    mask[kernel_half_height: mask_height - (kernel_half_height) - (kernel_height % 2),
         kernel_half_width: mask_width - (kernel_half_width) - (kernel_width % 2)] = image

    image_changes = np.zeros_like(mask)

    image_changes[kernel_half_height: image_height + kernel_half_height,
                    kernel_half_width + 1: image_width + kernel_half_width] += 1 * (image[:, 1:] == image[:, :-1])

    image_changes[kernel_half_height + 1: image_height + kernel_half_height,
                    kernel_half_width: image_width + kernel_half_width] = 1 * (image[1:, :] == image[:-1, :])

    dilation_mask = np.clip(image_changes, 0, 1.0)

    element = cv2.getStructuringElement(cv2.MORPH_RECT,  (kernel_width, kernel_height))
    dilation_mask = cv2.dilate(dilation_mask, element)

    for i in range(kernel_half_height, image_height + kernel_half_height):
        for j in range(kernel_half_width, image_width + kernel_half_width):
            if dilation_mask[i + kernel_half_height, j + kernel_half_width]:
                image_window = mask[i - kernel_half_height: i + (kernel_half_height) + (kernel_height % 2),
                                    j - kernel_half_width: j + (kernel_half_width) + (kernel_width % 2)]
                profile = image_window + kernel
                indices = np.unravel_index(np.argmax(profile, axis=None), profile.shape)
                ret_image[i - kernel_half_height, j - kernel_half_width] = image_window[indices] - (kernel_max - kernel[indices])
            else:
                ret_image[i - kernel_half_height, j - kernel_half_width] = mask[i, j]
    return ret_image

def add_shadow_artifacts(image, deg_start, direction=True, deg_spread=5):
    """
    Creates "shadows" in image
    :param image:
    :param deg_start: degree of shadow
    :param direction: direction of shadows (L or R)
    :param deg_spread: degree may be changed on every row
    :return: new image with artifacts
    """
    img_shape = np.shape(image)
    ret_image = np.zeros(img_shape)
    for i in range(0, img_shape[0]):
        degree = np.random.uniform(deg_start - deg_spread, deg_start + deg_spread)
        decrease = math.tan(degree * math.pi / 180) / 15
        prev = 0
        if direction:
            range_j = range(0, img_shape[1])
        else:
            range_j = range(img_shape[1] - 1, -1, -1)

        for j in range_j:
            ret_image[i, j] = image[i, j] if image[i, j] > prev - decrease else prev - decrease
            prev = ret_image[i, j]
    return ret_image


def apply_x_correlated_noise(gt, alpha, sigma, flip=False):
    noise = np.zeros_like(gt)

    norm_noise = sigma * np.random.randn(*(gt.shape))
    noise[..., 0] = norm_noise[..., 0]

    for j in range(1, gt.shape[-1]):
        noise[..., j] = alpha * noise[..., j - 1] + norm_noise[..., j]

    if flip:
        return gt + np.flip(noise, -1)

    return gt + noise


def grad_overshoot_markov(gt, t, mag, p_keep, p_weaken, weaken_factor, flip=False):
    if flip:
        gt = np.flip(gt, axis=-1)
    sobel_x = cv2.Sobel(gt, cv2.CV_64F, 1, 0, ksize=3)
    overshoot = np.where(np.abs(sobel_x) > t, sobel_x * mag, 0.0)

    r = np.random.rand(*(overshoot.shape))
    multiplier = np.where(r > p_weaken + p_keep, 0.0, 1.0)
    multiplier = np.where(r < p_weaken, weaken_factor, multiplier)

    for i in range(1, gt.shape[1]):
        overshoot[:, i] += overshoot[:, i - 1] * multiplier[:, i]

    if flip:
        return np.flip(gt + overshoot, axis=-1)

    return gt + overshoot


def add_linear_skew(img, sigma_a=0.2, sigma_b=0.2):
    x = np.linspace(0.0, 1.0, img.shape[1])
    y = np.linspace(0.0, 1.0, img.shape[0])
    mg_x, mg_y = np.meshgrid(x, y)
    a = sigma_a * np.random.randn()
    b = sigma_b * np.random.randn()

    return img + a * mg_x + b * mg_y


def add_parabolic_skew(img, sigma_a=0.2, sigma_b=0.2):
    c_x = np.random.rand()
    c_y = np.random.rand()

    a = sigma_a * np.random.randn()
    b = sigma_b * np.random.randn()

    x = np.linspace(0, 1.0, img.shape[1])
    y = np.linspace(0, 1.0, img.shape[0])
    mg_x, mg_y = np.meshgrid(x, y)

    return img + a * (mg_x - c_x) ** 2 + b * (mg_y - c_y) ** 2


if __name__ == '__main__':
    tips_path = 'D:/Research/data/GEFSEM/synth/res/tips.pkl'
    tips = load_tips_from_pkl(tips_path)
    tips_keys = list(tips.keys())

    images = []
    tips_generated = []

    for i in range(100):
        tip = np.random.choice(tips_keys)
        rot = np.random.randint(0, 3)
        tip_scaled = normalize(tips[tip]['data'])[:20, :]

        image = generate_grid_structure(256, 256)

        images.append(image)
        tips_generated.append(tip)

    start_time = time.time()
    for image, tip in zip(images, tips_generated):
        fast_dil_image = fast_dilate(image, tip_scaled)
        dil_image = dilate(image, tip_scaled)

        print(np.sum(np.abs(fast_dil_image - dil_image)))


