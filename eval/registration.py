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
from utils.image import normalize


def register_affine_orb(img_1, img_2, max_features=100):
    orb = cv2.ORB_create(max_features)

    (kpsA, descsA) = orb.detectAndCompute((255 * img_1).astype(np.uint8), None)
    (kpsB, descsB) = orb.detectAndCompute((255 * img_2).astype(np.uint8), None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA, descsB, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # matches = sorted(matches, key=lambda x: x.distance)

    # match_img = cv2.drawMatches(img_1, kpsA, img_2, kpsB, matches, None, flags=2)
    # cv2.imshow("Matches", match_img)
    # cv2.waitKey(0)

    if len(good) < 6:
        return None

    pts_1 = np.zeros((len(good), 2), dtype="float")
    pts_2 = np.zeros((len(good), 2), dtype="float")
    for (i, match) in enumerate(good):
        pts_1[i] = kpsA[match[0].queryIdx].pt
        pts_2[i] = kpsB[match[0].trainIdx].pt

    affine_matrix, mask = cv2.estimateAffinePartial2D(pts_1, pts_2)

    if np.abs(np.linalg.det(affine_matrix[:2, :2])) < 0.5:
        affine_matrix = None

    return affine_matrix


def command_iteration(method):
    if (method.GetOptimizerIteration() == 0):
        print("Estimated Scales: ", method.GetOptimizerScales())
    print(f"{method.GetOptimizerIteration():3} = {method.GetMetricValue():7.5f} : {method.GetOptimizerPosition()}")


def register_rigid_sitk(img_1, img_2, interpolator='linear', metric='mse', p1=None, p2=None, nOfBins=50, MMISamplingPercentage=0.5, verbose=True):
    img_1 = sitk.GetImageFromArray(img_1)
    img_2 = sitk.GetImageFromArray(img_2)

    if p1 is not None and p2 is not None:
        p1_list = [c for p in p1 for c in p]
        p2_list = [c for p in p2 for c in p]
        init_transform = sitk.LandmarkBasedTransformInitializer(sitk.Euler2DTransform(), p1_list, p2_list)
        # init_transform = sitk.LandmarkBasedTransformInitializer(sitk.AffineTransform(2), p1_list, p2_list)
    else:
        init_transform = sitk.Euler2DTransform()

        reg_init = sitk.ImageRegistrationMethod()
        reg_init.SetOptimizerAsExhaustive([32 // 2, 0, 0])
        reg_init.SetOptimizerScales([2.0 * np.pi / 32, 1.0, 1.0])

        reg_init.SetInitialTransform(sitk.CenteredTransformInitializer(img_1, img_2, init_transform))
        reg_init.SetInterpolator(sitk.sitkLinear)
        if verbose:
            reg_init.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(reg_init))
        init_transform = reg_init.Execute(img_1, img_2)

    reg = sitk.ImageRegistrationMethod()

    if metric == 'correlation':
        reg.SetMetricAsCorrelation()
    elif metric == 'mmi':
        reg.SetMetricAsMattesMutualInformation(nOfBins)
        reg.SetMetricSamplingPercentage(MMISamplingPercentage, sitk.sitkWallClock)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
    else:
        # defaults to mse
        reg.SetMetricAsMeanSquares()

    if interpolator == 'bspline':
        reg.SetInterpolator(sitk.sitkBSpline)
    else:
        # defaults to linear
        reg.SetInterpolator(sitk.sitkLinear)


    reg.SetOptimizerAsPowell(stepLength=0.1, numberOfIterations=250)
    # reg.SetOptimizerAsGradientDescent(learningRate=1.0e-3, numberOfIterations=1000, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(init_transform)
    reg.SetShrinkFactorsPerLevel(shrinkFactors=[16, 8, 4])
    reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if verbose:
        reg.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(reg))


    out_transform = reg.Execute(img_1, img_2)
    metric_value = reg.GetMetricValue()

    if verbose:
        print(out_transform)
        print("Metric: ", metric_value)

    return out_transform, metric_value


def resample_images(img_1, img_2, transform):
    if img_1.dtype == np.float32:
        img_1 = (255 * img_1).astype(np.uint8)
    if img_2.dtype == np.float32:
        img_2 = (255 * img_2).astype(np.uint8)

    img_1 = sitk.GetImageFromArray(img_1)
    img_2 = sitk.GetImageFromArray(img_2)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_1)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(100)
    resampler.SetTransform(transform)

    out = resampler.Execute(img_2)
    simg1 = sitk.Cast(sitk.RescaleIntensity(img_1), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    cimg = sitk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
    return normalize(sitk.GetArrayFromImage(simg2)), normalize(sitk.GetArrayFromImage(cimg))