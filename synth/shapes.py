import math
import random

import cv2
import numpy as np
from scipy import ndimage

from synth.imageSynthesizer import Synthesizer, params


def read_gray(deformation_file):
    img = cv2.imread(deformation_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray / 255.0


class ShapeClass():
    """
    Class representing different shapes added to generated images
    """
    def __init__(self, boundingBoxSize):
        """
        :param boundingBoxSize: every added object is in rectangle bounding box
        """
        self.x = boundingBoxSize[0]
        self.y = boundingBoxSize[1]
        self.shapeArr = None
        self.transforms = False
        self.normalize = self.normalize_inst
        self.resize = self.resize_inst
        self.inverted = False
        self.doNotResize = False

    def scaleHeight(self, factor):
        self.shapeArr *= factor

    def createPyramid(self, iterations):
        """
        Stacks objects on each other
        :param iterations: number of stacked objects
        :return: changes shapeArr
        """
        center = (self.x // 2, self.y // 2)

        resizeParams = (center[0] // iterations, center[1] // iterations)
        resizedArray = self.shapeArr
        x = self.x
        y = self.y
        for i in range(iterations):
            x = x - resizeParams[0]
            y = y - resizeParams[1]
            resizedArray = ShapeClass.resize(resizedArray, (x, y))
            self.mergeObjects(resizedArray)
        self.normalize_inst()

    def invert(self):
        self.inverted = not self.inverted
        self.shapeArr *= -1

    @staticmethod
    def resize(img, newSize):
        # return transform.resize(img, newSize, preserve_range=True)
        return cv2.resize(img, newSize)

    def resize_inst(self, newSize):
        self.x = newSize[0]
        self.y = newSize[1]
        self.shapeArr = ShapeClass.resize(self.shapeArr, newSize)

    def rotate(self, deg):
        self.shapeArr = ndimage.rotate(self.shapeArr, deg, reshape=True)
        self.x = np.shape(self.shapeArr)[0]
        self.y = np.shape(self.shapeArr)[1]

    def normalize_inst(self, arr=None):
        if arr is None:
            if self.shapeArr.min() == self.shapeArr.max():
                return
            self.shapeArr = (self.shapeArr - self.shapeArr.min()) / (self.shapeArr.max() - self.shapeArr.min())
        else:
            if arr.min() == arr.max():
                return
            return (arr - arr.min()) / (arr.max() - arr.min())

    @staticmethod
    def normalize(arr=None, distortion=0):
        if arr.min() == arr.max():
            return arr
        else:
            min = arr.min() + distortion if random.uniform(0,1) > 0.5 else arr.min()
            max = arr.max() - distortion if random.uniform(0,1) > 0.5 else arr.max()
            retval = (arr - min) / (max - min)
            return retval

    @staticmethod
    def addPixelNoise(image, range):
        mask = np.random.rand(np.shape(image)[0], np.shape(image)[1])
        mask = (range[1] - range[0]) * mask + range[0]
        image += mask
        return image

    def addBumps(self):
        """
        Add bumps to shapeArr
        :return:
        """
        syn = Synthesizer()
        syn.setParameters(params)
        filter = syn.createFilter(1)
        syn.addFilter(filter, self.shapeArr, noZero=True)
        self.shapeArr = Synthesizer.dilate(self.shapeArr, self.createGaussKernels(5, 1)[0])

    def rotateHeight(self, degX, degY):
        """
        Rotation along x and y axis
        :param degX:
        :param degY:
        :return:
        """
        tanX = math.tan(degX * math.pi / 180)
        tanY = math.tan(degY * math.pi / 180)
        for i in range(0,self.x):
            for j in range(0, self.y):
                self.shapeArr[i][j] += ((j + 1) * tanX) + ((i + 1) * tanY)
        self.shapeArr = (self.shapeArr - self.shapeArr.min()) / (self.shapeArr.max() - self.shapeArr.min())
        self.normalize_inst()

    # shapeObjMerge must have smaller or equal bounding box
    def mergeObjects(self, shapeObjMerge, inverse=False):
        """
        Merge shapeObjMerge into current object
        :param shapeObjMerge: object to merge
        :param inverse: invert shapeObjMerge
        :return: shapeArr is changed
        """
        if isinstance(shapeObjMerge, ShapeClass):
            r = int((self.x - shapeObjMerge.x) / 2)
            s = int((self.y - shapeObjMerge.y) / 2)
            if inverse:
                self.shapeArr[r:r + shapeObjMerge.x, s:s + shapeObjMerge.y] -= shapeObjMerge.shapeArr
            else:
                self.shapeArr[r:r + shapeObjMerge.x, s:s + shapeObjMerge.y] += shapeObjMerge.shapeArr

        else:
            x = np.shape(shapeObjMerge)[0]
            y = np.shape(shapeObjMerge)[1]
            r = int((self.x - x) / 2)
            s = int((self.y - y) / 2)
            if inverse:
                self.shapeArr[r:r + x, s:s + y] -= shapeObjMerge
            else:
                self.shapeArr[r:r + x, s:s + y] += shapeObjMerge


class Bubble(ShapeClass):
    """
    Bubble class
    """
    def __init__(self, bbox_size):
        super(Bubble, self).__init__(bbox_size)
        self.center = np.array([self.x / 2, self.y / 2])
        mg = np.mgrid[:self.x, :self.y]
        diffs = (mg - self.center[:, np.newaxis, np.newaxis])**2 / self.center[:, np.newaxis, np.newaxis]**2
        dist = np.sum(diffs, axis=0)
        height_sqr = 1.0 - dist ** 2
        height_sqr[height_sqr < 0.0] = 0.0
        self.shapeArr = np.sqrt(height_sqr)


class Rectangle(ShapeClass):
    """
    Rectangle class
    """
    def __init__(self, boundingBoxSize):
        super(Rectangle, self).__init__(boundingBoxSize)
        self.shapeArr = np.ones((boundingBoxSize[0], boundingBoxSize[1]))


class Barrel(ShapeClass):
    """
    Barrel class
    """

    def __init__(self, bbox_size):
        super(Barrel, self).__init__(bbox_size)
        self.shapeArr = np.zeros((bbox_size[0], bbox_size[1]))
        self.center = np.array([self.x / 2, self.y / 2])

        mg = np.mgrid[:self.x, :self.y]
        d = (mg - self.center[:, np.newaxis, np.newaxis]) ** 2 / self.center[:, np.newaxis, np.newaxis] ** 2

        self.shapeArr = 1 * (np.sum(d, axis=0) < 1.0)


class Deformation(ShapeClass):
    """
    Deformation class -> loaded from file
    """
    def __init__(self, deformationFile, boundingBoxSize):
        super(Deformation, self).__init__(boundingBoxSize)
        deformationArr = read_gray(deformationFile)
        self.shapeArr = ShapeClass.resize(deformationArr, boundingBoxSize)
        self.shapeArr = 1 * (self.shapeArr <= 0.9)