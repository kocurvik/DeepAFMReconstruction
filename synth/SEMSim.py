"""
SEM simulation from AFM
Algorthims are in more details here: https://www.researchgate.net/publication/34243893_Surface_reconstruction_from_AFM_and_SEM_images.
File includes simulation based on gradient and curvatures (not working well) and
Monte Carlo simulations -> working rather nicely. Advantage is, that we can control position of
detector here but gwyddion does not support it. There are more advanced SEM simulations on the internet
See https://www.gel.usherbrooke.ca/casino/
"""

import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.stats
from scipy import signal
import pickle
from scipy.optimize import minimize
import random
from scipy import interpolate
from scipy import ndimage
import time
import sys

class SEMSimulator:

    def __init__(self, params):
        self.afmImage = None
        self.semImage = None
        self.kernelSizes = params["kernelSizes"]
        self.saveFlag = params["save"]
        self.pathToFilters = params["pathToFilters"]
        #self.afmRes = np.shape(self.afmImage)
        self.filters = None
        self.divisors = None
        self.pathToWeights = params["pathToWeights"]
        if not os.path.exists(self.pathToWeights):
            self.weights0 = [0.01 for i in range(0,len(self.kernelSizes)*3 + 1)]
        else:
            with open(self.pathToFilters, 'rb') as f:
                self.weights0 = pickle.load(f)
                f.close()

        self.cropSize = params["cropSize"]
        self.mcLines = params["mcLines"]
        self.mcMuSigma = params["mcMuSigma"]
        self.mcScaling = params["mcScaling"]
        self.noiseLimit = params["noiseLimit"]

    def calculateFilters(self, neighbourhoodSizes, sigmas):
        """
        Calculates gaussian weighted filters for gradient and curvature convolutions.
        len(neighbourhoodSizes) == len(sigmas)
        :param neighbourhoodSizes: sizes of filters
        :param sigmas: sigma of gaussian wiegthing
        :return: pickle with these kernels
        """
        CsAll = []
        RsAll = []
        divisors = []
        kernelsAll = []
        sigmaIndex = 0
        for size in neighbourhoodSizes:
            pdfDist = scipy.stats.norm(0, sigmas[sigmaIndex])
            sigmaIndex += 1
            Rs = []
            Cs = []
            K5Divisor = 0
            for n in range(0, 4):
                for i in range(-(size // 2), size // 2):
                    for j in range(-(size // 2), size // 2):
                        K5Divisor += (i ** 2 * j ** 2)
                sum = 0
                for r in range(-(size // 2), size // 2):
                    weight = pdfDist.pdf(r)
                    sum += r ** (2 * n) * weight
                Rs.append(sum)
                sum = 0
                for c in range(-(size // 2), size // 2):
                    weight = pdfDist.pdf(c)
                    sum += c ** (2 * n) * weight
                Cs.append(sum)

            G = Rs[0] * Rs[2] * Cs[0] * Cs[2] - Rs[1] ** 2 * Cs[1] ** 2
            A = Rs[1] * Rs[3] * Cs[0] * Cs[2] - Rs[2] ** 2 * Cs[1] ** 2
            B = Rs[0] * Rs[2] * Cs[1] * Cs[3] - Rs[1] ** 2 * Cs[2] ** 2
            Q = Cs[0] * (Rs[0] * Rs[2] - Rs[1] ** 2)
            T = Rs[0] * (Cs[0] * Cs[2] - Cs[1] ** 2)
            U = Cs[0] * (Rs[1] * Rs[3] - Rs[2] ** 2)
            V = Cs[1] * (Rs[0] * Rs[2] - Rs[1] ** 2)
            W = Rs[1] * (Cs[0] * Cs[2] - Cs[1] ** 2)
            Z = Rs[0] * (Cs[1] * Cs[3] - Cs[2] ** 2)

            CsAll.append(Cs)
            RsAll.append(Rs)
            divisors.append(
                [1 / (Q * T), 1 / (U * W), 1 / (V * Z), 1 / Q, 1 / K5Divisor, 1 / T, 1 / U, 1 / V, 1 / W, 1 / Z])
            kernels = np.zeros((10, size, size))
            for r in range(-(size // 2), size // 2):
                for c in range(-(size // 2), size // 2):
                    weight = pdfDist.pdf(r) * pdfDist.pdf(c)
                    ri = r + size // 2
                    ci = c + size // 2
                    kernels[0, ri, ci] = (G - T * Rs[1] * r ** 2 - Q * Cs[1] * c ** 2) * weight
                    kernels[1, ri, ci] = (A - W * Rs[2] * r ** 2 - U * Cs[1] * c ** 2) * r * weight
                    kernels[2, ri, ci] = (B - Z * Rs[1] * r ** 2 - V * Cs[2] * c ** 2) * c * weight
                    kernels[3, ri, ci] = (Rs[0] * r ** 2 - Rs[1]) * weight
                    kernels[4, ri, ci] = r * c * weight
                    kernels[5, ri, ci] = (Cs[0] * c ** 2 - Cs[1]) * weight
                    kernels[6, ri, ci] = (Rs[1] * r ** 2 - Rs[2]) * r * weight
                    kernels[7, ri, ci] = (Rs[0] * r ** 2 - Rs[1]) * c * weight
                    kernels[8, ri, ci] = (Cs[0] * c ** 2 - Cs[1]) * r * weight
                    kernels[9, ri, ci] = (Cs[1] * c ** 2 - Cs[2]) * c * weight
            for i in range(0, 10):
                kernels[i] = np.flip(kernels[i], axis=0)
            kernelsAll.append(kernels)
        self.filters = kernelsAll
        self.divisors = divisors
        if self.saveFlag:
            with open(self.pathToFilters, 'wb') as f:
                pickle.dump({"filters" : self.filters, "divisors" : self.divisors}, f)
                f.close()

    def getNeighbourhood(self, image, centerCoords, size):
        """
        Gets neigbourhood of pixel
        :param image:
        :param centerCoords:
        :param size:
        :return:
        """
        imageX = np.shape(image)[0]
        imageY = np.shape(image)[1]
        kernelXDiv = size[0] // 2
        kernelYDiv = size[1] // 2

        paddedSize = (imageX + size[0], imageY + size[1])
        mask = np.zeros(paddedSize)
        mask[kernelXDiv: paddedSize[0] - (kernelXDiv) - (size[0] % 2),
        kernelYDiv: paddedSize[1] - (kernelYDiv) - (size[1] % 2)] = image

        nb = mask[centerCoords[1]:centerCoords[1]+size[0], centerCoords[1]:centerCoords[1]+size[1]]

        return nb

    def shuffleOrientation(self, points):
        """
        Some transformation from 1st quadrant only points to 4 quadrant points. Its random.
        :param points:
        :return: transformed points
        """
        selections = random.choices([0,1,2,3], k = len(points))
        newPoints = []
        for point, sel in zip(points, selections):
            if sel == 0:

                newPoints.append(point)
            elif sel == 1:

                newPoints.append([-point[0], point[1], point[2]])
            elif sel == 2:

                newPoints.append([point[0], -point[1], point[2]])
            elif sel == 3:

                newPoints.append([-point[0], -point[1], point[2]])
        return newPoints

    #DEPRECATED
    def fitPolynomial(self, nb):
        shapeNb = np.shape(nb)
        x = np.linspace(0 - shapeNb[0]//2, 1, 0 + shapeNb[0]//2)
        y = np.linspace(0 - shapeNb[1]//2, 1, 0 + shapeNb[1]//2)
        X, Y = np.meshgrid(x, y, copy=False)
        Z = X ** 2 + Y ** 2 + np.random.rand(*X.shape) * 0.01

        X = X.flatten()
        Y = Y.flatten()

        A = np.array([X * 0 + 1, X, Y, X ** 2, X ** 2 * Y, X ** 2 * Y ** 2, Y ** 2, X * Y ** 2, X * Y]).T
        B = Z.flatten()

        coeff, r, rank, s = np.linalg.lstsq(A, B)
        fitted_surf = np.polynomial.polynomial.polyval2d(x, y, coeff.reshape((4, 4)))
        plt.matshow(fitted_surf)

    #DEPRECATED
    def getInterpolator(self, nb):
        nbShape = (np.shape(nb))
        x = np.arange(-(nbShape[0] // 2), nbShape[0] // 2, 1)
        y = np.arange(-(nbShape[1] // 2), nbShape[1] // 2, 1)

        xx, yy = np.meshgrid(x, y)

        z = nb
        f = interpolate.interp2d(xx, yy, z, kind='linear')
        x2 = np.linspace(-(nbShape[0] // 2), nbShape[0] // 2, 1024)
        y2 = np.linspace(-(nbShape[1] // 2), nbShape[1] // 2, 1024)
        Z2 = f(x2, y2)

        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].pcolormesh(xx, yy, z)

        X2, Y2 = np.meshgrid(x2, y2)
        ax[1].pcolormesh(X2, Y2, Z2)
        plt.show()
        return f

    def mcMethod2D(self):
        """
        Simulates SEM with 2D MC method.
        :return: simulated SEM.
        """
        print("starting MC method 2D")
        simulatedImage = np.zeros((self.afmRes[0], self.afmRes[1]))
        img = self.normalize(self.afmImage)
        errorArr = np.asarray([i*(1/self.errSize) for i in range(0, self.errSize)])
        for i in range(0, self.afmRes[0]):
            for j in range(0, self.afmRes[1]):
                t1 = time.time()
                planeDistances = self.mcMuSigma * np.sqrt(-np.log(1.0 - np.random.uniform(0,1,self.mcLines)))
                #planeDistances = np.random.normal(self.mcMuSigma[0], self.mcMuSigma[1], self.mcLines)
                # planeDistances = np.multiply(cosins, linesLen)
                orientations = np.random.uniform(0, 2 * np.pi, self.mcLines)
                xs = np.cos(orientations) * planeDistances
                ys = np.sin(orientations) * planeDistances
                t2 = time.time()
                print("Time Pass1 : ", t2 - t1)
                points = [[x, y, z] for x, y, z in zip(xs, ys, [img[i,j] for k in range(0, self.mcLines)])]
                points = np.asarray(self.shuffleOrientation(points))
                fraction = 0
                t3 = time.time()
                print("Time Pass2 : ", t3 - t2)
                cmpi = np.round(i + points[:, 1])
                cmpj = np.round(j + points[:, 0])
                cmpi[cmpi >= self.afmRes[0]] = self.afmRes[0] - 1
                cmpj[cmpj >= self.afmRes[1]] = self.afmRes[1] - 1
                cmpi[cmpi <= 0] = 0
                cmpj[cmpj <= 0] = 0
                # vals = ndimage.map_coordinates(self.normalize(self.afmImage), [xAbs, yAbs], order=2)
                z0 = img[i,j]

                for c in range(0, self.mcLines):
                    z = img[int(cmpi[c]), int(cmpj[c])]
                    if z >= z0:
                        ss = -errorArr[round((z - z0)*self.errSize)]
                    else:
                        ss = errorArr[round((z - z0) * self.errSize)]
                    fraction += ss

                simulatedImage[i, j] = fraction / self.mcLines

        self.showImage(simulatedImage)
        self.saveImg(simulatedImage, os.path.join("nenovisionTasks", "simulatedImg2D.jpg"))
        return


    def getPoints2D(self, nOfPoints, i, j, angleRange, image):
        """
        Returns set of points (x,y,z) around (i,j) from image, used to simulate SEM signal.
        z is taken from image.
        :param nOfPoints: number of returned points
        :param i: x of center point
        :param j: y of center point
        :param angleRange:
        :param image:
        :return: arr of points
        """
        planeDistances = self.mcMuSigma * np.sqrt(-np.log(1.0 - np.random.uniform(0, 1, nOfPoints)))
        orientations = np.random.uniform(0,2*np.pi, nOfPoints)
        xs = np.cos(orientations) * planeDistances
        ys = np.sin(orientations) * planeDistances
        points = np.asarray([(x, y, 0) for x, y in zip(xs, ys)])
        points[:, 1] = np.round(i + points[:, 1])
        points[:, 0] = np.round(j + points[:, 0])
        points = np.asarray(points[(points[:, 1] > 0) & (points[:, 0] > 0) & (points[:, 0] < self.afmRes[0] - 1) & (
                points[:, 1] < self.afmRes[1] - 1)])

        heights = np.asarray([image[int(point[1]),int(point[0])] for point in points])
        points[:, 2] = heights

        return points

    def getPoints3D(self, nOfPoints, i, j, angleRange):
        """
        Returns set of points (x,y,z) from image. z is also simulated
        :param nOfPoints: number of returned points
        :param i: x of center point
        :param j: y of center point
        :param angleRange: angles of simulated points
        :return: arr of points
        """
        upAngles = np.random.uniform(0, 0.25 * np.pi, nOfPoints)
        tangs = np.tan(upAngles)
        planeDistances = self.mcMuSigma * np.sqrt(-np.log(1.0 - np.random.uniform(0, 1, nOfPoints)))
        # planeDistances = np.random.normal(0, self.mcMuSigma[1], self.mcLines)

        heights = (np.multiply(tangs, planeDistances) + self.afmImage[i, j]) * self.mcScaling
        orientations = np.random.uniform(angleRange[0], angleRange[1], nOfPoints)
        xs = np.cos(orientations) * planeDistances
        ys = np.sin(orientations) * planeDistances
        points = np.asarray([(x, y, z) for x, y, z in zip(xs, ys, heights)])
        points[:, 1] = np.round(i + points[:, 1])
        points[:, 0] = np.round(j + points[:, 0])
        points = np.asarray(points[(points[:, 1] > 0) & (points[:, 0] > 0) & (points[:, 0] < self.afmRes[0] - 1) & (
                    points[:, 1] < self.afmRes[1] - 1)])
        return points

    def mcMethod3D(self, image, angleRange, dimensions=2):
        """
        2D should work better. Difference is, that 3D also simulates height of returned electron.
        :param image: AFM image
        :param angleRange: angles used to simulate height of backscattered electron
        :param dimensions:
        :return: 3D simulated image
        """
        self.afmImage = image
        self.afmRes = np.shape(self.afmImage)
        print("starting MC method 3D")
        simulatedImage = np.zeros((self.afmRes[0], self.afmRes[1]))

        img = self.normalize(self.afmImage)

        for i in range(0, self.afmRes[0]):
            for j in range(0, self.afmRes[1]):
                '''
                colors = np.array(self.normalize(np.asarray(points)[:,2]) * 100)

                plt.scatter(np.asarray(points)[:,0], np.asarray(points)[:,1], c = colors, cmap = 'viridis')
                plt.colorbar()
                plt.show()
                '''
                iter = 0
                ss = 0
                ss2 = 0
                while True:
                    if dimensions == 2:
                        points = self.getPoints2D(self.mcLines, i, j, angleRange, self.afmImage)
                        z = np.asarray([img[i,j] for c in range(0, np.shape(points)[0])])
                    else:
                        points = self.getPoints3D(self.mcLines, i, j, angleRange)
                        z = np.asarray(
                            [img[int(points[c, 1]), int(points[c, 0])] for c in range(0, np.shape(points)[0])])

                    iter += np.shape(points)[0]

                    s = np.sum(z - points[:,2] )
                    ss += s
                    ss2 += s * s
                    mean = ss / iter
                    disp = (ss2 / iter) - (mean * mean)

                    mean = 0.5 * (1.0 + mean)
                    disp /= 2.0 * iter
                    a = self.noiseLimit * mean * (1.0 - mean)

                    if (disp < self.noiseLimit * mean * (1.0 - mean)) or i < 5 or i > self.afmRes[0] -5 or j < 5 or j > self.afmRes[1] - 5:
                        a = 2
                        break

                simulatedImage[i,j] = np.sum(ss) / iter if iter != 0 else 0
        #self.showImage(self.afmImage)
        simulatedImage = self.normalize(simulatedImage)
        self.showImage(simulatedImage[5:self.afmRes[0] - 5, 5:self.afmRes[1] - 5])
        #self.saveImg(simulatedImage, os.path.join("nenovisionTasks", "simulatedImg3D.jpg"))
        return simulatedImage

    def showImage(self, data):
        plt.figure()
        plt.imshow(data, interpolation='none', origin='upper')
        plt.show()

    def normalize(self, arr=None):
        return (arr - arr.min()) / (arr.max() - arr.min())

    def simulateImage(self, weights):
        """
        Returns SEM image simulated by filter bank model specified by learned weights.
        :param weights: Weights we look for.
        :return: simulated image
        """
        img = weights[0] + np.sum(np.multiply(weights[1:len(self.kernelSizes) + 1], self.gradientMagnitude) \
                                  + np.multiply(weights[len(self.kernelSizes) + 1:len(self.kernelSizes) * 2 + 1],
                                                self.maximumCurvature) \
                                  + np.multiply(weights[len(self.kernelSizes) * 2 + 1:len(self.kernelSizes) * 3 + 1],
                                                self.minimumCurvature), axis=2)
        return img


    def costFunction(self, weights):
        """
        Error function that is used as an input for optimizer.
        :param weights: Array of weights used.
        :return: Scalar value of error function.
        """
        img = weights[0] + np.sum(np.multiply(weights[1:len(self.kernelSizes) + 1], self.gradientMagnitude) \
                                  + np.multiply(weights[len(self.kernelSizes) + 1:len(self.kernelSizes)*2 + 1], self.maximumCurvature) \
                                   + np.multiply(weights[len(self.kernelSizes)*2 + 1:len(self.kernelSizes)*3 + 1], self.minimumCurvature), axis=2)
        d = np.sum(np.abs(weights[0] + np.sum(np.multiply(weights[1:len(self.kernelSizes) + 1], self.gradientMagnitude) \
                                              + np.multiply(weights[len(self.kernelSizes) + 1:len(self.kernelSizes)*2 + 1], self.maximumCurvature) \
                                   + np.multiply(weights[len(self.kernelSizes)*2 + 1:len(self.kernelSizes)*3 + 1], self.minimumCurvature), axis=2) - self.semImage))
        return d

    def filterImage(self):
        """
        Filter AFM image stored in instance attribute using filter bank model. !!! CHANGE SOME PATHS
        :return: Saves simulated SEM to path (hard coded).
        """
        # if conv filters for gradient and curvature exist, we do not calculate them
        if (not os.path.exists(self.pathToFilters) or self.saveFlag):
            self.calculateFilters(self.kernelSizes, self.mcMuSigma)
        if self.filters is None or self.divisors is None:
            with open(self.pathToFilters, 'rb') as f:
                dict = pickle.load(f)
                self.filters = dict['filters']
                self.divisors = dict['divisors']
                f.close()
        gradientMagnitude = np.zeros((self.afmRes[0], self.afmRes[1], len(self.kernelSizes)))
        minimumCurvature = np.zeros((self.afmRes[0], self.afmRes[1], len(self.kernelSizes)))
        maximumCurvature = np.zeros((self.afmRes[0], self.afmRes[1], len(self.kernelSizes)))
        convolvedWithK = np.zeros((self.afmRes[0], self.afmRes[1], len(self.kernelSizes), len(self.filters[0])))

        for nSize in range(0, len(self.kernelSizes)):
            for j in range(1,6):
                convolvedWithK[:,:,nSize,j] = signal.convolve2d(self.afmImage,self.filters[nSize][j], 'same') * self.divisors[nSize][j]
            gradientMagnitude[:,:,nSize] = np.sqrt((np.square(convolvedWithK[:,:,nSize,1]) + np.square(convolvedWithK[:,:,nSize,2])))
            self.showImage(gradientMagnitude[:,:,nSize])
            maximumCurvature[:,:,nSize] = convolvedWithK[:,:,nSize,5] + convolvedWithK[:,:,nSize,3] + \
                                          np.sqrt(np.square(convolvedWithK[:,:,nSize,5] - convolvedWithK[:,:,nSize,3]) + \
                                                  np.square(convolvedWithK[:,:,nSize,4]))
            self.showImage(maximumCurvature[:, :, nSize])
            minimumCurvature[:,:,nSize] = convolvedWithK[:,:,nSize,5] + convolvedWithK[:,:,nSize,3] - \
                                          np.sqrt(np.square(convolvedWithK[:,:,nSize,5] - convolvedWithK[:,:,nSize,3]) + \
                                                  np.square(convolvedWithK[:,:,nSize,4]))
            self.showImage(minimumCurvature[:, :, nSize])
            self.saveImg(gradientMagnitude[:,:,nSize], os.path.join("nenovisionTasks", str(nSize) + "g.jpg"))
            self.saveImg(maximumCurvature[:, :, nSize], os.path.join("nenovisionTasks", str(nSize) + "a.jpg"))
            self.saveImg(minimumCurvature[:,:,nSize], os.path.join("nenovisionTasks", str(nSize) + "b.jpg"))
        cs = self.cropSize
        self.gradientMagnitude = gradientMagnitude[cs:-cs,cs:-cs,:]
        self.minimumCurvature = minimumCurvature[cs:-cs,cs:-cs,:]
        self.maximumCurvature = maximumCurvature[cs:-cs,cs:-cs,:]
        self.semImage = self.normalize(self.semImage[cs:-cs,cs:-cs])

        for i in range(0,len(self.kernelSizes)):
            self.gradientMagnitude[:, :, i] = self.normalize(self.gradientMagnitude[:,:,i])
            self.minimumCurvature[:, :, i] = self.normalize(self.minimumCurvature[:, :, i])
            self.maximumCurvature[:, :, i] = self.normalize(self.maximumCurvature[:, :, i])
        print("Initial Cost Function: ", self.costFunction(self.weights0))
        res = minimize(self.costFunction, np.asarray(self.weights0), method='Nelder-Mead',options={'xatol': 1e-8, 'disp': True})
        if self.saveFlag:
            with open(self.pathToWeights, 'wb') as f:
                pickle.dump(res.x, f)
                f.close()
        print(res)
        img = self.simulateImage(res.x)
        imgInitial = self.simulateImage(self.weights0)
        self.showImage(imgInitial)
        self.showImage(self.semImage)
        self.saveImg(img, os.path.join("nenovisionTasks", "simulatedImgFB.jpg"))
        print("asdf")

    def saveImg(self, arr, path):
        arr = (self.normalize(arr) * 255)
        im1 = Image.fromarray(arr)
        im1 = im1.convert('L')
        im1.save(path)

    def testFunc(self):
        img = np.asarray(Image.open(os.path.join("nenovisionTasks", "simulatedImg.jpg")))
        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(img, kernel, iterations=1)
        self.showImage(erosion)

params = {
        "afm" : os.path.join("nenovisionTasks", "4a.png"),
        "sem" : os.path.join("nenovisionTasks", "4b.png"),
        "pathToFilters" : 'filters.pkl', # precalculated filters for filter bank model (weighted functions of filter size)
        "pathToWeights" : 'weights.pkl',# precalculated weights for filter bank model
        "kernelSizes" : [5,7,9,11,15,21,29], # sizes of filters
        #"kernelSizes" : [5,7,9,11],
        "sigmas" : [0.6,0.8485,1.2,1.697,2.4,3.394,4.8], # sigmas of wweighting function
        #"sigmas" : [0.6,0.8485,1.2,1.697],
        "save" : True, # flag if we want to save filters and weights
        "cropSize" : 15, # bound crop
        "mcLines" : 20, # number of monte carlo simulated electrons
        "mcMuSigma" : (12), # sigma of gaussian lenght of line
        "mcScaling" : 0.0039, # bulgarian constant for scaling
        "noiseLimit" : 1e-3 # no idea what this does
}


