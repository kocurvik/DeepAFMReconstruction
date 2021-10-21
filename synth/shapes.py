import math
import random

import cv2
import numpy as np
from scipy import ndimage

# from synth.imageSynthesizer import Synthesizer, params
from utils.image import normalize


def read_gray(deformation_file):
    img = cv2.imread(deformation_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray / 255.0


def create_grid(arr, repeat_y, repeat_x, padding_y, padding_x):
    height = arr.shape[0]
    width = arr.shape[1]
    new_arr = np.zeros([height + 2 * padding_y, width + 2 * padding_x])
    new_arr[padding_y: new_arr.shape[0] - padding_y, padding_x: new_arr.shape[1] - padding_x] = arr

    grid = np.tile(arr, [repeat_y, repeat_x])

    return grid


def create_pyramid(arr, iterations, height_offset, width_offset, resize_method=cv2.INTER_NEAREST):
    height = arr.shape[0]
    width = arr.shape[1]

    if height - 2 * iterations * height_offset < 1 or width - 2 * iterations * width_offset < 1:
        raise(ValueError("Too many iterations in pyramid construction!"))

    for i in range(1, iterations + 1):
        start_y = i * height_offset
        start_x = i * width_offset
        end_y = height - i * height_offset
        end_x = width - i * width_offset
        new_height = end_y - start_y
        new_width = end_x - start_x
        new_arr = cv2.resize(arr, (new_width, new_height), interpolation=resize_method)
        arr[start_y: end_y, start_x: end_x] += new_arr
    return arr


def create_bubble_array(height, width):
    mg = np.mgrid[:height, :width]
    center = np.array([height / 2, width / 2])

    diffs = (mg - center[:, np.newaxis, np.newaxis]) ** 2 / center[:, np.newaxis, np.newaxis] ** 2
    dist = np.sum(diffs, axis=0)
    height_sqr = 1.0 - dist ** 2
    height_sqr[height_sqr < 0.0] = 0.0
    return np.sqrt(height_sqr)


def create_barrel_array(height, width):
    mg = np.mgrid[:height, :width]
    center = np.array([height / 2, width / 2])
    d = (mg - center[:, np.newaxis, np.newaxis]) ** 2 / center[:, np.newaxis, np.newaxis] ** 2
    return 1.0 * (np.sum(d, axis=0) < 1.0)


def create_plane_array(height, width, rot_x=0.0, rot_y=0.0):
    if rot_x == 0.0 and rot_y == 0.0:
        return np.zeros([height, width])

    mg_y, mg_x = np.mgrid[:height, :width].astype(np.float32)
    mg_y /= height
    mg_x /= width

    tan_x = np.tan(rot_x)
    tan_y = np.tan(rot_y)

    return tan_y * mg_y + tan_x * mg_x

#
# class Deformation(ShapeClass):
#     """
#     Deformation class -> loaded from file
#     """
#     def __init__(self, deformationFile, boundingBoxSize):
#         super(Deformation, self).__init__(boundingBoxSize)
#         deformationArr = read_gray(deformationFile)
#         self.shapeArr = ShapeClass.resize(deformationArr, boundingBoxSize)
#         self.shapeArr = 1 * (self.shapeArr <= 0.9)


if __name__ == '__main__':
    cv2.imshow("Plane", create_plane_array(100, 100))
    cv2.imshow("Plane - rot", create_plane_array(100, 100, rot_x=0.1, rot_y=0.7))
    cv2.imshow("Barrel", create_barrel_array(100, 200))
    cv2.imshow("Bubble", create_bubble_array(350, 200))
    cv2.imshow("Pyramid", normalize(create_pyramid(create_barrel_array(200, 200), 3, 5, 10)))
    cv2.imshow("Grid", create_grid(create_bubble_array(10, 14), 3, 8, 2, 0))
    cv2.waitKey(0)
