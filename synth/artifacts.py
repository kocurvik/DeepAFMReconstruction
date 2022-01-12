import math
import time

import cv2
import numpy as np

from synth.generator import generate_grid_structure
from synth.tip_dilation import dilate, fast_dilate
from utils.image import load_tips_from_pkl, normalize


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

    tips_array = np.array([tips[k]['data'] for k in tips_keys])

    m_tips = np.max(tips_array, axis=(-1, -2))

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




class Artifactor():
    def __init__(self, **kwargs):
        for (prop, default) in Artifactor.get_default_param_dict().items():
            setattr(self, prop, kwargs.get(prop, default))

    @staticmethod
    def get_default_param_dict():
        default_params = {
            # skew params
            'linear_skew_sigma': 0.1, 'parabolic_skew_sigma': 0.1, 'skew_prob': 0.25,

            # overshoot params
            'overshoot_prob': 0.5, 'max_overshoot_t': 0.5, 'max_overshoot_mag': 0.2, 'min_p_keep': 0.0,
            'max_p_keep': 0.9, 'min_weaken_factor': 0.0, 'max_weaken_factor': 0.5,

            # shadows params
            'shadows_prob': 0.5, 'shadows_uniform_p': 0.5, 'shadows_uniform_both_p': 0.5, 'shadows_randomize_prob': 0.5,
            'shadows_max': 1.5, 'shadows_max_randomize_percentage': 0.3,

            # x-correlated noise params
            'noise_prob': 0.8, 'noise_alpha_min': 0.00, 'noise_alpha_max': 0.9, 'noise_sigma_min': 0.0001,
            'noise_sigma_max': 0.1}
        return default_params

    def add_overshoot(self, image, flip=False):
        t = np.random.uniform(0, self.max_overshoot_t)
        mag = np.random.uniform(0, self.max_overshoot_mag)
        p_keep = np.random.uniform(self.min_p_keep, self.max_p_keep)
        p_weaken = np.random.uniform(0.0, 1 - p_keep - 0.05)
        weaken_factor = np.random.uniform(self.min_weaken_factor, self.max_weaken_factor)
        image = grad_overshoot_markov(image, t, mag, p_keep, p_weaken, weaken_factor, flip=flip)

        return image

    def add_skew(self, image):
        image = add_linear_skew(image, sigma_a=self.linear_skew_sigma, sigma_b=self.linear_skew_sigma)
        image = add_parabolic_skew(image, sigma_a=self.parabolic_skew_sigma, sigma_b=self.parabolic_skew_sigma)
        return image

    def add_shadows(self, image, flip=False, per_pixel_decrease=None):
        if per_pixel_decrease is None:
            per_pixel_decrease = 1 / np.random.uniform(0, self.shadows_max * image.shape[1])

        if np.random.rand() < self.shadows_randomize_prob:
            max_randomize_percentage = np.random.uniform(0, self.shadows_max_randomize_percentage)

            per_pixel_decrease += per_pixel_decrease * np.random.uniform(-max_randomize_percentage, max_randomize_percentage, image.shape[0])

        if flip:
            image = np.flip(image, axis=-1)

        prev = np.zeros(image.shape[0])
        for i in range(image.shape[1]):
            image[:, i] = np.where(image[:, i] > prev, image[:, i], prev)
            prev = image[:, i] - per_pixel_decrease

        if flip:
            image = np.flip(image, axis=-1)
        return image

    def add_noise(self, image, flip=False):
        sigma = np.random.uniform(self.noise_sigma_min, self.noise_sigma_max)
        alpha = np.random.uniform(self.noise_alpha_min, self.noise_alpha_max)
        return apply_x_correlated_noise(image, alpha, sigma, flip=flip)

    def apply(self, img):
        img_l = np.copy(img)
        img_r = np.copy(img)

        if np.random.rand() < self.overshoot_prob:
            img_l = self.add_overshoot(img_l)
            img_r = self.add_overshoot(img_r, flip=True)

        if np.random.rand() < self.skew_prob:
            img_l = self.add_skew(img_l)
            img_r = self.add_skew(img_r)

        if np.random.rand() < self.shadows_prob:
            if np.random.rand() < self.shadows_uniform_both_p:
                per_pixel_decrease = 1 / np.random.uniform(0, self.shadows_max * img_l.shape[1])
            else:
                per_pixel_decrease = None
            img_l = self.add_shadows(img_l, per_pixel_decrease=per_pixel_decrease)
            img_r = self.add_shadows(img_r, flip=True, per_pixel_decrease=per_pixel_decrease)

        if np.random.rand() < self.noise_prob:
            img_l = self.add_noise(img_l)
            img_r = self.add_noise(img_r, flip=True)

        return img_l, img_r


