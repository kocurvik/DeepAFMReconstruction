import cv2
import numpy as np

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


class FastTipDilator():
    def __init__(self, tips_path, **kwargs):
        if tips_path is None:
            return
        tips_dict = load_tips_from_pkl(tips_path)
        self.tips = np.array([normalize(v['data']) for k, v in tips_dict.items()])

        for (prop, default) in FastTipDilator.get_default_param_dict().items():
            setattr(self, prop, kwargs.get(prop, default))

    @staticmethod
    def get_default_param_dict():
        default_params = {'tip_scale_min': 1.0, 'tip_scale_max': 50.0}
        return default_params

    def get_random_tip(self):
        tip = self.tips[np.random.choice(len(self.tips))]
        rot = np.random.randint(0, 3)
        scale = np.random.uniform(self.tip_scale_min, self.tip_scale_max)
        tip_scaled = scale * tip
        for _ in range(0, rot):
            tip_scaled = np.rot90(tip_scaled)
        return tip_scaled

    def apply(self, img_gt):
        tip = self.get_random_tip()
        return fast_dilate(img_gt, tip)