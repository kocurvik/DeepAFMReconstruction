'''
Author: Juraj Skandera
Date: 22.3.2021

Description: Class for simple registration with default parameters.
Simple ITK supports many options for registration. This script serves as a skeleton for more options.
Based on previous experiments, default transformation is set as a translation, optimizer is gradient descent with regular step.
For single modal registration, metric is correlation, for multimodal, Mattes mutual information is used.
Single modal registration should work well, multimodal depends on the input.

Usage:
   reg = Registration(params=defaultParams)
   reg.register()
'''
import copy
import math

import SimpleITK as sitk
import os

import cv2
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import skimage
from scipy import ndimage
from scipy import signal
from skimage.metrics import structural_similarity as ssim
from skimage import feature
# usage:
#   reg = Registration()
#   reg.register()

# default parameters for class init

# IN CASE OF AFM/SEM REGISTRATION, MOVING IMAGE MUST BE AFM
metrics = {"Correlation" : 1, "MMI" : 2, "JHMI" : 3}
defaultParams = {
    "fixedImage" : os.path.join("nenovisionTasks", "3a.png"),
    "movingImage" : os.path.join("nenovisionTasks", "3b.png"),
    "transformFile" : os.path.join("nenovisionTasks", "transform.txt"),
    "outputPathFixed" :  os.path.join("nenovisionTasks", "out1.png"),
    "outputPathMoving" :  os.path.join("nenovisionTasks", "out2.png"),
    "composePath" : os.path.join("nenovisionTasks", "composite.png"),
    "format" : "jpg",

    "metric" : metrics["Correlation"],
    "MMISamplingPercentage" : 0.5,
    "nOfBins" : 50,

    "maxIter" : 1000,
    "gradientTolerance" : 1e-5,
    "learningRate" : 0.0005,
    "convergenceMin" : 1e-5,

}



class Registration():
    """
    Facade class for sitk image registration. Also supports registration based on common points in image.
    """
    def __init__(self, params=defaultParams):
        format = params["format"]
        if format == "png":
            self.readFormatParam = "PNGImageIO"
        elif format == "jpeg":
            self.readFormatParam = "JPEGImageIO"
        else:
            self.readFormatParam = "BMPImageIO"

        self.transformPath = params["transformFile"]
        self.outputPathFixed = params["outputPathFixed"]
        self.outputPathMoving = params["outputPathMoving"]
        self.fixedImagePath = params["fixedImage"]
        self.movingImagePath = params["movingImage"]
        self.composePath = params["composePath"]
        self.transformation = params["transformation"]

        self.fixedImage = sitk.ReadImage(params["fixedImage"], sitk.sitkFloat32)
        self.movingImage = sitk.ReadImage(params["movingImage"], sitk.sitkFloat32)
        self.regMethod = sitk.ImageRegistrationMethod()

        if params["metric"] == 1:
            self.regMethod.SetMetricAsCorrelation()
        elif params["metric"] == 2:
            self.regMethod.SetMetricAsMattesMutualInformation(params["nOfBins"])
            self.regMethod.SetMetricSamplingPercentage(params["MMISamplingPercentage"], sitk.sitkWallClock)
            self.regMethod.SetMetricSamplingStrategy(self.regMethod.RANDOM)
        elif params["metric"] == 3:
            self.regMethod.SetMetricAsJointHistogramMutualInformation(params["nOfBins"])

        self.regMethod.SetOptimizerAsRegularStepGradientDescent(params["learningRate"], params["convergenceMin"], params["maxIter"], params["gradientTolerance"])
        if self.transformation == "translation":
            initTransform = sitk.TranslationTransform(self.fixedImage.GetDimension())
            self.regMethod.SetInitialTransform(initTransform)
        elif self.transformation == "spline":
            transformDomainMeshSize = [2] * self.fixedImage.GetDimension()
            tx = sitk.BSplineTransformInitializer(self.fixedImage,
                                                  transformDomainMeshSize)
            self.regMethod.SetInitialTransformAsBSpline(tx,
                                           inPlace=True,
                                           scaleFactors=[1, 2, 5])
            self.regMethod.SetShrinkFactorsPerLevel([4, 2, 1])
            self.regMethod.SetSmoothingSigmasPerLevel([4, 2, 1])

        self.regMethod.AddCommand(sitk.sitkIterationEvent, lambda: self.commandIteration())

    def findTransform(self, im1, im2):
        """
        Grid search of transformation based on structural similarity
        :param im1: fixed image
        :param im2: moving image
        :return: transform
        """
        paramsGrid = {"tX": [-2, -1, 0, 1, 2], "tY": [-2, -1, 0, 1, 2], "rot": [-0.5, -0.25, 0, 0.25, 0.5], "shear" : [-0.04, 0.0, 0.04]}
        finalTrans = skimage.transform.AffineTransform(rotation=math.radians(0), translation=(0,0))
        maxCoeff = 0
        print(np.shape(im1))
        print(np.shape(im2))
        print("SSIM Coeff: ", ssim(im1, im2))
        #self.showImage(feature.canny(im2, sigma=4).astype(int) - feature.canny(im1, sigma=4).astype(int))
        for i in paramsGrid["tX"]:
            for j in paramsGrid["tY"]:
                for k in paramsGrid["rot"]:
                    for s in paramsGrid["shear"]:

                        initialTransform = skimage.transform.AffineTransform(rotation=math.radians(k), translation=(i,j), shear=s)
                        arrMoving = skimage.transform.warp(im2, inverse_map=initialTransform)
                        coeff = ssim(im1, arrMoving)
                        #self.showImage(arrMoving)
                        if coeff > maxCoeff:
                            print(coeff, " ", initialTransform.rotation, " ", initialTransform.translation, " ", initialTransform.shear)
                            finalTrans = copy.deepcopy(initialTransform)
                            maxCoeff = coeff
        arrMoving = skimage.transform.warp(im2, inverse_map=finalTrans)
        print(self.fixedImagePath)
        print(self.movingImagePath)
        print("Final Registration")
        #self.showImage(im1 - arrMoving)
        #self.showImage(feature.canny(im1, sigma=4).astype(int) - feature.canny(arrMoving, sigma=4).astype(int))
        return finalTrans

    def commandIteration(self):
        if (self.regMethod.GetOptimizerIteration() == 0):
            print("Estimated Scales: ", self.regMethod.GetOptimizerScales())
        print("{0:3} = {1:7.5f}".format(self.regMethod.GetOptimizerIteration(),
                                              self.regMethod.GetMetricValue()))

    def smallestbox(self, a):
        r = a.any(1)
        if r.any():
            m, n = a.shape
            c = a.any(0)
            out = a[r.argmax():m - r[::-1].argmax(), c.argmax():n - c[::-1].argmax()]
        else:
            out = np.empty((0, 0), dtype=bool)
        return out

    def showImage(self, data):
        plt.figure()
        plt.imshow(data, interpolation='none', origin='upper')
        plt.show()

    def normalize(self, arr):
        return (arr - arr.min()) / (arr.max() - arr.min())

    def saveImg(self, arr, path):
        arr = (self.normalize(arr) * 255)
        im1 = Image.fromarray(arr)
        im1 = im1.convert('L')
        im1.save(path)

    def registerSimple(self, initialPoints=None, transforms=None):
        """
        Registration based on initial common points in image and grid search of advanced registration.
        Its controlled by attributes sets in init.
        :param initialPoints: common points in image
        :param transforms: if set, transforms are not calculated but these are used instead (initial, precision)
        :return: initial point to point transforms, precision transforms from grid search
        """
        src= initialPoints[0]
        dst = initialPoints[1]
        #src = np.array([0, 0, 10, 10]).reshape((2, 2))
        #dst = np.array([12, 14, 1, -20]).reshape((2, 2))
        if transforms is None:
            initialTransform = skimage.transform.estimate_transform('affine', src, dst)
        else:
            initialTransform = transforms[0]

        arrMoving = sitk.GetArrayFromImage(self.movingImage)
        arrMoving[arrMoving==0] = 50
        arrMoving = skimage.transform.warp(arrMoving, inverse_map=initialTransform)
        arrFixed = sitk.GetArrayFromImage(self.fixedImage)
        arrFixed[arrFixed==0] = 50
        mask = self.createMask2(arrFixed,arrMoving)
        arrMoving = self.maskImage(arrMoving, mask)
        arrFixed = self.maskImage(arrFixed, mask)
        #arrFixed = self.smallestbox(arrFixed)
        #arrMoving = self.smallestbox(arrMoving)
        #self.movingImage = sitk.GetImageFromArray(arrMoving)
        #self.fixedImage = sitk.GetImageFromArray(arrFixed)
        #x = self.smallestbox(arrMoving)
        #self.showImage(arrMoving)
        if transforms is None:
            precisionTrans = self.findTransform(arrFixed, arrMoving)
        else:
            precisionTrans = transforms[1]

        arrMoving = skimage.transform.warp(arrMoving, inverse_map=precisionTrans)
        mask = self.createMask2(arrFixed, arrMoving)
        arrMoving = self.maskImage(arrMoving, mask)
        arrFixed = self.maskImage(arrFixed, mask)
        self.saveImg(arrFixed, self.outputPathFixed)
        self.saveImg(arrMoving, self.outputPathMoving)

        return initialTransform, precisionTrans

    def maskImage(self, arr, mask):
        newArr = np.zeros(np.shape(mask))
        arrShape = np.shape(arr)
        newArr[0:arrShape[0],0:arrShape[1]] = arr
        arr = newArr*mask
        return arr

    def createMask(self,arrFixed, arrMoving):
        shape1 = np.shape(arrFixed)
        shape2 = np.shape(arrMoving)
        shape = (shape1[0] if shape1[0] > shape2[0] else shape2[0],shape1[1] if shape1[1] > shape2[1] else shape2[1])
        mask = np.zeros(shape)
        mask[0:shape1[0],0:shape1[1]] = arrFixed
        mask[mask>0] = 1
        return mask

    def createMask2(self, arrFixed, arrMoving):
        shape1 = np.shape(arrFixed)
        shape2 = np.shape(arrMoving)
        shape = (shape1[0] if shape1[0] > shape2[0] else shape2[0], shape1[1] if shape1[1] > shape2[1] else shape2[1])
        mask = np.zeros(shape)
        mask[0:shape2[0], 0:shape2[1]] = arrMoving
        mask[mask > 0] = 1
        return mask

    def register(self, initialPoints=None, transforms=None):
        """
        sitk registration
        :param initialPoints: if set, initial point to point transform is calculates
        :param transforms: precalculated transforms
        :return: saves registered images
        """
        print(self.outputPathFixed, " ",self.outputPathMoving)
        initialTransform = None
        if initialPoints is not None:
            try:
                initialTransform = cv2.estimateAffinePartial2D(np.asarray(initialPoints["semPts"]), np.asarray(initialPoints["afmPts"]))[0]
                #initialTransform = np.asarray([[1,0,initialTransform[0,2]], [0,1, initialTransform[1,2]]])
            except Exception as e:
                print(e)
            arrMoving = sitk.GetArrayFromImage(self.movingImage)
            arrFixed = sitk.GetArrayFromImage(self.fixedImage)
            if initialTransform is not None:
                self.movingImage = cv2.warpAffine(arrMoving, initialTransform, np.shape(arrFixed))
                self.movingImage = sitk.GetImageFromArray(self.movingImage)

        arrMoving = sitk.GetArrayFromImage(self.movingImage)
        arrFixed = sitk.GetArrayFromImage(self.fixedImage)
        if transforms is None:
            outTx = self.regMethod.Execute(self.fixedImage, self.movingImage)
        else:
            outTx = transforms[2]

        #self.regMethod.SetOptimizerScalesFromIndexShift()
        #sitk.WriteTransform(outTx, self.transformPath + str(uuid.uuid1()) + '.txt')

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(self.fixedImage)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(outTx)

        out = resampler.Execute(self.movingImage)

        simg1 = sitk.Cast(sitk.RescaleIntensity(self.fixedImage), sitk.sitkUInt8)
        simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)

        arrFixed = sitk.GetArrayFromImage(simg1)
        arrMoving = sitk.GetArrayFromImage(simg2)
        if self.transformation != 'spline':
            mask = self.createMask2(arrFixed,arrMoving)
            arrFixed = self.maskImage(arrFixed, mask)
            arrMoving = self.maskImage(arrMoving,mask)
            arrFixed = self.smallestbox(arrFixed)
            arrMoving = self.smallestbox(arrMoving)
        print(np.shape(arrFixed), " ", np.shape(arrMoving))
        cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(self.composePath)
        writer.Execute(cimg)

        self.saveImg(arrFixed, self.outputPathFixed)
        self.saveImg(arrMoving, self.outputPathMoving)
        return outTx