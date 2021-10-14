"""
File contains classes and functions used for synthetic data generation and some AD HOC functions used for some experiments
Data generation is controlled by parameters passed to init.
Usage:
    syn = Synthesizer(params)
    syn.generateData()
"""
import math
from scipy.ndimage import gaussian_filter
import imageio
# from DataManipulator import DataManipulator
import gwyfile
from PIL import ImageFilter
import copy

# from Registration import Registration, defaultParams
from synth.shapes import ShapeClass, Bubble, Rectangle, Barrel, Deformation
from SEMSim import *

# params used for SEM simulation
paramsSEMSim = {
    "afm" : os.path.join("nenovisionTasks", "4a.png"),
    "sem" : os.path.join("nenovisionTasks", "4b.png"),
    "pathToFilters" : 'filters.pkl',
    "pathToWeights" : 'weights.pkl',
    "kernelSizes" : [5,7,9,11,15,21,29],
    #"kernelSizes" : [5,7,9,11],
    "sigmas" : [0.6,0.8485,1.2,1.697,2.4,3.394,4.8],
    #"sigmas" : [0.6,0.8485,1.2,1.697],
    "save" : True,
    "cropSize" : 0,
    "mcLines" : 20,
    "mcMuSigma" : (5),
    "mcScaling" : 0.0039,
    "noiseLimit" : 0.005
}

params = {
    'filters' : [{
        'mu' : 0,
        'sigma' : 10,
        'size' : 15,
        'coverage' : 200,
        'scaling' : 1
    },
    {
        'mu' : 0,
        'sigma' : 5,
        'size' : 10,
        'coverage' : 200,
        'scaling' : 1
    }
    ],
    'filterTypes' : ['Gaussian', 'Gaussian'],
    'blurSigma' : 1,
    'deformations' : ['deformation1.jpg', 'deformation2.jpg', 'deformation3.jpg'],
    'kernelFiles' : ['tips.gwy']
}


class Synthesizer():
    """
    Synthesizer class. For params description, see default params dict in the end of source file
    """
    def __init__(self, params):
        self.index = 0
        self.objNMin = params["objNMin"]
        self.objNMax = params["objNMax"]
        self.generatorStep = params["generatorStep"]
        self.datasetLen = params["datasetLen"]
        self.npyOrImages = params["npyOrImages"]
        self.shiftProb = params["shiftProb"]
        self.shiftBound = params["shiftBound"]
        self.gridPercentage = params["gridPercentage"]
        self.dilation = params["dilation"]
        self.tipsPath = params["tipsPath"]
        self.pathToRealData = params["pathToRealData"]
        self.dataFolder = params['dataFolder']
        self.filterParams = None
        self.filterTypes = None
        self.blurSigma = None
        self.mode = params["mode"]
        self.blur = params["blur"]
        self.gaussNoise = params["gaussNoise"]
        self.blurDataPercentage = params["blurDataPercentage"]
        self.noisePercentage = params["noisePercentage"]
        self.noiseParams = params["noiseParams"]
        self.pathToBackgrounds = params["pathToBackgrounds"]
        self.simulator = None
        self.SEMSim = params["SEMSim"]
        self.resolution = params["resolution"]
        self.bumps = params["bumps"]

    @staticmethod
    def rotate(image, deg):
        return ndimage.rotate(image, deg, reshape=True)

    @staticmethod
    def normalize(arr=None, distortion=0):
        if arr.max() == arr.min():
                return arr /1
        return (arr - arr.min()) / (arr.max() + distortion - arr.min())

    def setParameters(self, params):
        """
        Sets parameters for blurring and bumps
        :param params:
        :return:
        """
        self.filterParams = params['filters']
        self.filterTypes = params['filterTypes']
        self.nOfFilters = len(self.filterTypes)
        self.blurSigma = params['blurSigma']

    # @staticmethod
    def createFilter(self, i):
        """
        Creates filter used for bumps
        :param i:
        :return:
        """
        currFil = {}
        size = np.random.randint(2,12)
        mu = 0
        sigma = np.random.uniform(15,20)
        scaling = 1
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        d = np.sqrt(x * x + y * y)
        g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        g = g * scaling
        g = ShapeClass.normalize(g)
        currFil['filter'] = g
        currFil['coverage'] = np.random.uniform(20,50)
        return currFil

    def addFilter(self, filter, image, noZero=False):
        """
        Filters image with filter
        :param filter:
        :param image:
        :param noZero:
        :return: filtered image
        """
        resIm = np.shape(image)
        resFil = np.shape(filter['filter'])
        sizeIm = resIm[0] * resIm[1]
        sizeFil = resFil[0] * resFil[1]
        numberOfApps = int((sizeIm / sizeFil) * (filter['coverage'] / 100))
        a = np.expand_dims(np.random.randint(0, resIm[0], numberOfApps), axis=1)
        b = np.expand_dims(np.random.randint(0, resIm[1], numberOfApps), axis=1)
        locations = np.concatenate((a,b), axis=1)
        for location in locations:
            u = 0
            for i in range(location[0], location[0] + resFil[0] if location[0] + resFil[0] < resIm[0] else  resIm[0]):
                v = 0
                for j in range(location[1], location[1] + resFil[1] if location[1] + resFil[1] < resIm[1] else  resIm[1]):
                    image[i][j] = image[i][j] + filter['filter'][u][v] if image[i][j] != 0 else 0
                    v += 1
                u += 1
        return image

    def showImage(self, data):
        fig = plt.figure()
        imgplot = plt.imshow(data, interpolation='none', origin='upper')
        plt.show()

    def createPlane(self, resolution):
        plane = np.ones((resolution[0], resolution[1]))
        return plane

    def createImage(self, resolution):
        plane = self.createPlane([resolution[0], resolution[1]])
        for i in range(0, 1):
            filter = self.createFilter(i)
            plane = self.addFilter(filter, plane)
        plane = gaussian_filter(plane, sigma=self.blurSigma)
        plane = (plane - plane.min()) / (plane.max() - plane.min())

        return plane

    @staticmethod
    def gaussFilter(img):
        return gaussian_filter(img, params['blurSigma'])

    @staticmethod
    def scale(img, factor):
        return img * factor

    @staticmethod
    def addObject(shapeObj, image, pos, additive=False, relativeSize=None):
        """
        Adds object to image on position
        :param shapeObj: object to add
        :param image: obejct is added here
        :param pos: position of added object
        :param additive: add or replace
        :param relativeSize:
        :return: image with added object
        """
        imX = np.shape(image)[0]
        imY = np.shape(image)[1]
        xRange = None
        yRange = None
        imPosX = int(pos[0] - shapeObj.x / 2)
        if imPosX < 0:
            xStart = -imPosX
            imPosX = 0
        else:
            xStart = 0

        imPosY = int(pos[1] - shapeObj.y / 2)
        if imPosY < 0:
            yStart = -imPosY
            imPosY = 0
        else:
            yStart = 0


        xRange = shapeObj.x if (imPosX + shapeObj.x) < imX else (imX - imPosX)

        yRange = shapeObj.y if (imPosY + shapeObj.y) < imY else (imY - imPosY)

        iteratorX = imPosX
        for x in range(xStart, xRange):
            iteratorY = imPosY
            for y in range(yStart, yRange):
                if not additive:
                    if (shapeObj.shapeArr[x][y] < 0.01 and shapeObj.shapeArr[x][y] > -0.01):
                        image[iteratorX][iteratorY] = image[iteratorX][iteratorY]
                    else:
                        image[iteratorX][iteratorY] = shapeObj.shapeArr[x][y]
                else:
                    image[iteratorX][iteratorY] = image[iteratorX][iteratorY] if (
                                shapeObj.shapeArr[x][y] < 0.0001 and shapeObj.shapeArr[x][y] > -0.0001) \
                        else image[iteratorX][iteratorY] + shapeObj.shapeArr[x][y]
                iteratorY += 1
            iteratorX += 1
        return image

    def randomImage(self, resolution, objects):
        """
        Creates new base image without artifacts
        :param resolution:
        :param objects: all objects that can be added
        :return: new image
        """
        image = np.zeros((resolution, resolution))
        # add rotated plane background
        if random.uniform(0,1) < 0.1:
            background = Rectangle((resolution, resolution))
            rotDegX = random.uniform(0,15)
            rotDegY = random.uniform(0,15)
            background.rotateHeight(rotDegX, rotDegY)
            rotDeg = random.randint(0,4) * 90
            background.rotate(rotDeg)
            background.resize((resolution, resolution))
            image = Synthesizer.addObject(background, image, (resolution//2,resolution//2), additive=True)
        # add background from file
        if random.uniform(0,1) < 0.05:
            background = Rectangle((resolution, resolution))
            image = random.choice(self.backgrounds)
            background.shapeArr = image
            rotDeg = random.randint(0,4) * 90
            background.rotate(rotDeg)
            background.resize((resolution, resolution))
            image = Synthesizer.addObject(background, image, (resolution // 2, resolution // 2), additive=True)
        # control intensity of background w.r. to objects -> images in step 1 have dark bckg, step 2 is more bright
        if self.generatorStep == 2:
            image += random.uniform(-0.5, 0.5)


        minObj = paramsSyn["objNMin"]
        maxObj = paramsSyn["objNMax"]

        nOfObjects = random.randint(minObj,maxObj)

        for i in range(0, nOfObjects):
            index = random.randint(0, len(objects) - 1)
            maxSize = resolution
            shapeObj = objects[index]
            newObj = None
            if not isinstance(shapeObj, Rectangle):
                newSize = (random.randint(1, maxSize), random.randint(1, maxSize))
                newObj = copy.deepcopy(shapeObj)
                newObj.resize(, newSize
                if self.bumps < np.random.uniform(0,1):
                    filter = syn.createFilter(np.random.uniform(0, 3))
                    self.addFilter(filter, newObj.shapeArr, noZero=True)

                if self.generatorStep == 2 and random.uniform(0,1) > 0.3 and newObj.inverted is False:
                    newObj.invert()
                newObj.shapeArr *= random.uniform(0,1)
            image = Synthesizer.addObject(newObj if newObj is not None else shapeObj, image, (random.randint(0, resolution), random.randint(0, resolution)), additive=True)

        image = ShapeClass.normalize(image)

        return image

    def createGrid(self, resolution, sizesPercent, rotDeg):
        """
        Created grid
        :param resolution:
        :param sizesPercent: sizes of edges of rectangles in grid
        :param rotDeg: rotation degree
        :return: grid
        """
        size1 = int((resolution / 100) * sizesPercent[0])
        size2 = int((resolution / 100) * sizesPercent[1])

        rec1 = Rectangle((size1, size1))
        rec1.scaleHeight(40)

        rec2 = Rectangle((size2, size1))
        rec2.scaleHeight(18)

        rec3 = Rectangle((size1, size2))
        rec3.scaleHeight(14)

        rec4 = Rectangle((size2, size2))

        image = np.zeros((resolution, resolution))

        crop = {}
        yCoord = 0
        col = 1
        while yCoord < resolution + size1:
            xCoord = 0
            while xCoord < resolution + size1:
                if col % 2:
                    image = Synthesizer.addObject(rec1, image, (xCoord, yCoord))
                    xCoord = xCoord + size1 // 2 + size2 // 2
                    image = Synthesizer.addObject(rec2, image, (xCoord, yCoord))
                    xCoord = xCoord + size1 // 2 + size2 // 2
                else:
                    image = Synthesizer.addObject(rec3, image, (xCoord, yCoord))
                    xCoord = xCoord + size1 // 2 + size2 // 2
                    image = Synthesizer.addObject(rec4, image, (xCoord, yCoord))
                    xCoord = xCoord + size1 // 2 + size2 // 2
            yCoord = yCoord + size1 // 2 + size2 // 2
            col += 1
        image = Synthesizer.normalize(image)

        #image = Synthesizer.rotate(image, rotDeg)
        # image = Synthesizer.gaussFilter(image)
        crop['S'] = 0
        crop['E'] = resolution
        cropCoord = resolution

        if rotDeg != 0:
            for i in range(0, resolution):
                if math.fabs(image[0][i]) > 0.01 or math.fabs(image[1][i]) > 0.01:
                    cropCoord = i
            resNew = np.shape(image)[0]
            crop['E'] = cropCoord
            crop['S'] = resNew - cropCoord
        image = image[crop['S']:crop['E'], crop['S']:crop['E']]
        image = ShapeClass.resize(image, (resolution, resolution))
        #self.saveImg(image, 'img_notDilates.jpg')
        return image

    def createObjects(self):
        """
        Creates many different shape objects that are used during generation
        :return: arr of shape objects
        """
        objects = []

        for i in range(1,5):
            barrel = Barrel((50, 50))
            barrel.invert()
            barrel.doNotResize = True
            objects.append(barrel)

        for i in range(1,12):
            for j in range(0,4):
                deform = Deformation(os.path.join('deformations', 'deformation' + str(i) + '.jpg'), (100, 100))
                if j != 0:
                    deform.createPyramid(j)
                if random.uniform(0,1) < 0.5:
                    deform.rotate(45)
                objects.append(deform)

        barrel2 = Barrel((100, 100))
        barrel2.invert()
        barrel2.mergeObjects(Barrel((50, 50)))
        objects.append(barrel2)

        barrel2 = Barrel((100, 100))
        barrel2.mergeObjects(Barrel((50, 50)), True)
        objects.append(barrel2)

        objects.append(Bubble((100, 100)))

        bubble2 = Bubble((100, 100))
        bubble2.invert()
        bubble2.mergeObjects(Bubble((50, 50)), True)
        objects.append(bubble2)

        bubble3 = Bubble((100, 100))
        objects.append(bubble3)


        line = Rectangle((100, 3))
        line.rotate(15)
        objects.append(line)

        line = Rectangle((100, 2))
        line.rotate(45)
        objects.append(line)
        rectangle = Rectangle((512, 50))
        rectangle.rotateHeight(20,20)
        objects.append(rectangle)

        rectangle = Rectangle((100, 100))
        rectangle.mergeObjects(Bubble((50, 50)))
        objects.append(rectangle)

        rectangle = Rectangle((100, 100))
        rectangle.createPyramid(3)
        rectangle.shapeArr += 1
        objects.append(rectangle)

        rectangle = Rectangle((100, 100))
        rectangle.rotateHeight(30,0)
        rectangle.rotate(45)
        objects.append(rectangle)

        for object in objects:
            if not object.doNotResize:
                object.normalize_inst()

        return objects

    def showImage(self, data):
        plt.figure()
        plt.imshow(data, interpolation='none', origin='upper')
        plt.show()

    def erode(self, image, kernel):
        """
        Erodes image by kernel
        :param image:
        :param kernel:
        :return:
        """
        kernel = np.flip(kernel, 0)
        kernel = np.flip(kernel, 1)
        imageX = np.shape(image)[0]
        imageY = np.shape(image)[1]
        kernelX = np.shape(kernel)[0]
        kernelY = np.shape(kernel)[1]
        kernelXDiv = kernelX // 2
        kernelYDiv = kernelY // 2

        maskShapeX = imageX + kernelX
        maskShapeY = imageY + kernelY
        mask = np.ones((maskShapeX, maskShapeY))
        mask[kernelXDiv: maskShapeX - (kernelXDiv) - (kernelX % 2),
        kernelYDiv: maskShapeY - (kernelYDiv) - (kernelY % 2)] = image
        newImage = np.zeros((imageX, imageY))
        kernelMaxInd = np.unravel_index(np.argmax(kernel, axis=None), kernel.shape)

        for i in range(kernelXDiv, imageX + kernelXDiv):
            for j in range(kernelYDiv, imageY + kernelYDiv):
                partOfImage = mask[i - kernelXDiv: i + (kernelXDiv) + (kernelX % 2),
                              j - kernelYDiv: j + (kernelYDiv) + (kernelY % 2)]
                profile = partOfImage - kernel
                indices = np.unravel_index(np.argmin(profile, axis=None), profile.shape)
                newImage[i - kernelXDiv, j - kernelYDiv] = partOfImage[indices] + kernel[kernelMaxInd] - kernel[indices]

        return newImage

    @staticmethod
    def dilate(image, kernel):
        """
        Dilates image by kernel
        :param image:
        :param kernel:
        :return:
        """
        imageX = np.shape(image)[0]
        imageY = np.shape(image)[1]
        kernelX = np.shape(kernel)[0]
        kernelY = np.shape(kernel)[1]
        kernelXDiv = kernelX // 2
        kernelYDiv = kernelY // 2

        maskShapeX = imageX + kernelX
        maskShapeY = imageY + kernelY
        mask = np.zeros((maskShapeX, maskShapeY))
        mask[kernelXDiv: maskShapeX - (kernelXDiv) - (kernelX % 2), kernelYDiv: maskShapeY - (kernelYDiv) - (kernelY % 2)] = image
        newImage = np.zeros((imageX, imageY))
        kernelMaxInd = np.unravel_index(np.argmax(kernel, axis=None), kernel.shape)

        for i in range(kernelXDiv, imageX + kernelXDiv):
            for j in range(kernelYDiv, imageY + kernelYDiv):
                partOfImage = mask[i - kernelXDiv: i + (kernelXDiv) + (kernelX % 2), j - kernelYDiv: j + (kernelYDiv) + (kernelY % 2)]
                profile = partOfImage + kernel
                indices = np.unravel_index(np.argmax(profile, axis=None), profile.shape)
                newImage[i - kernelXDiv, j - kernelYDiv] = partOfImage[indices] - (kernel[kernelMaxInd] - kernel[indices])

        return newImage

    def createArtifacts(self, image, degreeStart, direction=True, degreeSpread=5):
        """
        Creates "shadows" in image
        :param image:
        :param degreeStart: degree of shadow
        :param direction: direction of shadows (L or R)
        :param degreeSpread: degree may be changed on every row
        :return: new image with artifacts
        """
        imgShape = np.shape(image)
        newImg = np.zeros(imgShape)
        for i in range(0, imgShape[0]):
            degree = np.random.uniform(degreeStart - degreeSpread, degreeStart + degreeSpread)
            decrease = math.tan(degree * math.pi / 180) / 15
            prev = 0
            if direction:
                rangeJ = range(0, imgShape[1])
            else:
                rangeJ = range(imgShape[1] - 1, -1, -1)

            for j in rangeJ:
                newImg[i][j] = image[i][j] if image[i][j] > prev - decrease else prev - decrease
                prev = newImg[i][j]
        return newImg

    def getKernels(self, size):
        """
        DEPRECATED
        :param size:
        :return:
        """
        print("Loading kernels")
        directory = os.path.join('allGWYFiles','tips')
        nKernelFiles = len(os.listdir(directory))
        kernels = []
        for i in os.listdir(directory):
            kernelFile = i
            channels = gwyfile.util.get_datafields(gwyfile.load(os.path.join(directory, kernelFile)))
            keys = list((channels.keys()))
            for channel in keys:
                data = np.reshape(channels[channel]['data'],
                                                         (channels[channel]['xres'], channels[channel]['yres']))
                kernel = ShapeClass.normalize(ShapeClass.resize(data, (size, size)))
                kernel = Synthesizer.normalize(kernel)
                kernels.append(kernel)
        return kernels

    def createErrorSignal(self, image, direction):
        """
        Creates useless error signal of image. Can be rewrited as convolution
        :param image:
        :param direction:
        :return:
        """
        imgShape = np.shape(image)
        newImg = np.zeros(imgShape)

        for i in range(0, imgShape[0]):
            if direction:
                rangeJ = range(0, imgShape[1])
            else:
                rangeJ = range(imgShape[1] - 1, -1, -1)

            prev = 0
            for j in rangeJ:
                if (j == 0 or i == 0 and direction) or (j == max(rangeJ) or i == imgShape[0] and not direction):
                    newImg[i][j] = 0
                else:
                    newImg[i][j] = image[i][j] - prev
                prev = image[i][j]
        return newImg


    def shiftHist(self, left, right, distRange):
        """
        Shifts intensity by distRange but bounds stay same
        :param left: image with left shadows
        :param right: image with right shadows
        :param distRange: max shift
        :return: left, right and one is shifted
        """
        dist = random.uniform(0, distRange)
        ch = random.choice([True, False])

        if ch:
            left = left + dist
            left[left>1] = 1
        else:
            right = right + dist
            right[right>1] = 1
        return left, right

    def addNoise(self, a,b):
        """
        Adds AFM specific noise to images correlated in x direction
        :param a: L image
        :param b: R image
        :return: L, R image
        """
        left = copy.deepcopy(a)
        right = copy.deepcopy(b)

        std = np.random.uniform(self.noiseParams[0], self.noiseParams[1])
        alpha = np.random.uniform(self.noiseParams[2], self.noiseParams[3])
        shape = np.shape(left)
        currentValL = 0.0
        currentValR = 0.0
        for i in range(0, shape[0]):
            for j in range(0, shape[1]):
                currentValL = alpha * currentValL + np.random.normal(0, std)
                left[i,j] += currentValL
                currentValR = alpha * currentValR + np.random.normal(0, std)
                right[i,j] += currentValR

        return self.normalize(left), self.normalize(right)


    def saveImg(self, arr, path):
        arr = (arr * 255)
        im1 = Image.fromarray(arr)
        im1 = im1.convert('L')
        im1.save(path)

    def blurImage(self, arr, radius):
        """
        Simple blur
        :param arr:
        :param radius:
        :return: blurred image
        """
        im = (ShapeClass.normalize(arr) * 255)
        im = Image.fromarray(im)
        im = im.convert('L')
        im = im.filter(ImageFilter.GaussianBlur(radius=radius))
        arr2 = self.normalize(np.asarray(im))
        return arr2

    def saveNewImgs(self, kernel, image, angles):
        """
        Creates and saves synthetic image.
        :param kernel: kernel for dilation (tip)
        :param image: ground truth synthesized image
        :param angles: angles of shadows if we generate shadow data
        :return: saves data to specified folder
        """
        global paramsSEMSim
        semImage = None

        image = self.blurImage(image,0.5)
        if self.blur and random.uniform(0,1) < self.blurDataPercentage:
            image = self.blurImage(image,2)


        dilImage = image
        if self.simulator is not None:
            start = random.uniform(0,math.pi * 2)
            diff = random.uniform(0, math.pi * 2)
            semImage = self.simulator.mcMethod3D(dilImage, (start, diff + start))


        if random.uniform(0,1) < self.dilation:
            dilImage = self.dilate(dilImage, kernel)
        if random.uniform(0, 1) < self.noisePercentage and self.mode == 3:
            dilImage, _ = self.addNoise(dilImage, dilImage)
        path = os.path.join(self.dataFolder, 'train')
        path2 = os.path.join(self.dataFolder, 'train', 'label')

        if self.mode == 2:
            for angle in angles:
                filteredL = dilImage
                filteredR = dilImage

                if random.uniform(0, 1) < self.noisePercentage:
                    filteredL, filteredR = self.addNoise(filteredL, filteredR)
                self.index += 1
                filteredR = self.createArtifacts(filteredR, angle, False, 0)
                filteredL = self.createArtifacts(filteredL, angle, True, 0)

                if self.npyOrImages == 'images':
                    pathImages = os.path.join(self.dataFolder,'Images')
                    if not os.path.exists(pathImages):
                        os.mkdir(pathImages)
                    self.saveImg(filteredL, os.path.join(pathImages, '{:05d}'.format(self.index)) + "L.png")
                    self.saveImg(filteredR, os.path.join(pathImages, '{:05d}'.format(self.index)) + "R.png")
                    self.saveImg(image, os.path.join(pathImages, '{:05d}'.format(self.index)) + ".png")
                    if semImage is not None:
                        self.saveImg(semImage, os.path.join(pathImages, str(self.index)) + "SEM.png")
                else:
                    with open(os.path.join(path,str(self.index)), 'wb') as f:
                        if self.mode == 1:
                            arr = np.array([image])
                        else:
                            arr = np.array([filteredL, filteredR])
                        arr = np.swapaxes(arr, 0,2)
                        arr = np.swapaxes(arr, 0,1)
                        np.save(f, arr)
                    with open(os.path.join(path2, str(self.index)), 'wb') as f:
                        np.save(f, np.array([image]))
        else:
            #if random.uniform(0, 1) < self.burstNoiseProb:
            #    dilImage, _ = self.addBurstNoise(dilImage, None, self.burstNoisePercentage)
            self.index += 1
            if self.npyOrImages == 'images':
                pathImages = os.path.join(self.dataFolder, 'images')
                if not os.path.exists(pathImages):
                    os.mkdir(pathImages)
                self.saveImg(dilImage, os.path.join(pathImages, 'train', str(self.index) + ".png"))
                self.saveImg(image, os.path.join(pathImages, 'train', 'label', str(self.index)) + ".png")
                self.saveImg(np.abs(dilImage - image), os.path.join(pathImages, 'train', 'label', str(self.index)) + "DIFF.png")
                if semImage is not None:
                    self.saveImg(semImage, os.path.join(pathImages, 'train', 'label', str(self.index)) + "SEM.png")
                if self.mode == 3:
                    self.saveImg(kernel, os.path.join(pathImages, 'train', 'label', str(self.index) + 'TIP.png'))
            else:
                with open(os.path.join(path, str(self.index)), 'wb') as f:
                    arr = np.array([dilImage])
                    arr = np.swapaxes(arr, 0, 2)
                    arr = np.swapaxes(arr, 0, 1)
                    np.save(f, arr)
                with open(os.path.join(path2, str(self.index)), 'wb') as f:
                    np.save(f, np.array([image]))



    def createGaussKernels(self, size, n, double=False):
        """
        Creates gaussian kernels.
        :param size: size of kernel
        :param n: number of kernels
        :param double: if true, one different kernel is generated
        :return:
        """
        kernels = []
        mu = 0
        sigma = 0.5
        if not double:
            for i in range(0,n):
                x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
                d = np.sqrt(x * x + y * y)
                g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
                g = Synthesizer.normalize(g)
                kernels.append(g)
                sigma += 0.2
        else:
            x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
            d = np.sqrt(x * x + y * y)
            g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            g = Synthesizer.normalize(g)
            sizeNew = size // 2
            x, y = np.meshgrid(np.linspace(-1, 1, sizeNew), np.linspace(-1, 1, sizeNew))
            d = np.sqrt(x * x + y * y)
            g2 = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
            g2 = Synthesizer.normalize(g2)
            g2 = g2 * 0.6
            g[size - sizeNew:size, sizeNew//4:sizeNew//4 + sizeNew] += g2
            g = Synthesizer.normalize(g)
            kernels.append(g)
        return kernels

    def kernelsToPickle(self):
        """
        Creates few gaussian kernels
        :return:
        """
        kernels1 = self.createGaussKernels(50, 3)
        kernels = self.getKernels((50, 50))
        kernels.extend(kernels1)

        filehandler = open('syntheticData\\kernels.pkl', "wb")
        pickle.dump(kernels, filehandler)
        filehandler.close()

    def getGTAndTips(self, pathToRealPickle):
        """
        DEPRECATED
        :param pathToRealPickle:
        :return:
        """
        dataFile = open(pathToRealPickle, 'rb')
        data = pickle.load(dataFile)

        gtAndTips = {}
        for fileKey in data.keys():
            channels = data[fileKey]
            gtFlag = False
            gtAndTips[fileKey] = {}
            for channel in channels:
                if 'Estimated tip' in channel:
                    gtAndTips[fileKey]['tip'] = data[fileKey][channel]
                if '_GT' in channel:
                    gtAndTips[fileKey]['gt'] = data[fileKey][channel]
                    gtFlag = True
                if 'Topo' in channel and not gtFlag:
                    gtAndTips[fileKey]['gt'] = data[fileKey][channel]
        keys = list(gtAndTips.keys())
        for fileKey in keys:
            if len(gtAndTips[fileKey]) != 2:
                gtAndTips.pop(fileKey)
        dataFile.close()
        return gtAndTips

    def addToPickle(self, pathToRealPickle, filesNew):
        """
        Adds new files to existing pickle. Way to expand newRealData.pkl
        :param pathToRealPickle:
        :param filesNew:
        :return:
        """
        with open(pathToRealPickle, "rb") as f:
            gwyFiles = pickle.load(f)
            gwyFileNames = gwyFiles.keys()
            for fileName in filesNew:
                fileName = fileName.split('.')[0]
                searchedGwyFile = [x for x in gwyFileNames if x.split('.')[0] in fileName]
                if len(searchedGwyFile) > 0:
                    searchedGwyFile = searchedGwyFile[0]
                else:
                    continue
                realResKey = list(gwyFiles[searchedGwyFile].keys())[0]
                realRes = (gwyFiles[searchedGwyFile][realResKey]['xreal'], gwyFiles[searchedGwyFile][realResKey]['yreal'])
                gwyFiles[searchedGwyFile][fileName] = {}
                gwyFiles[searchedGwyFile][fileName]['data'] = filesNew[fileName]
                gwyFiles[searchedGwyFile][fileName]['xres'] = np.shape(gwyFiles[searchedGwyFile][fileName]['data'])[0]
                gwyFiles[searchedGwyFile][fileName]['yres'] = np.shape(gwyFiles[searchedGwyFile][fileName]['data'])[1]
                gwyFiles[searchedGwyFile][fileName]['xreal'] = realRes[0]
                gwyFiles[searchedGwyFile][fileName]['yreal'] = realRes[1]
            f.close()
        f = open(pathToRealPickle, "wb")
        pickle.dump(gwyFiles, f)
        f.close()
        print("Modified data added, Pickle file closed")

    def simulateRealData(self, pathToRealPickle, tipsPath):
        """
        DEPRECATED
        :param pathToRealPickle:
        :param tipsPath:
        :return:
        """
        tipsFile = open(tipsPath, 'rb')
        tips = pickle.load(tipsFile)
        tips = tips[list(tips.keys())[0]]
        tipsKeys = list(tips.keys())
        gtAndTips = self.getGTAndTips(pathToRealPickle)
        changedChannels = {}
        for fileKey in gtAndTips.keys():
            channelNameFstPart = fileKey.split('.')[0]
            gtTip = gtAndTips[fileKey]['tip']['data']
            gt = gtAndTips[fileKey]['gt']['data']
            kernel = gtTip
            gtDilated = self.dilate(gt, kernel)
            changedChannels[channelNameFstPart + '_DilGT'] = gtDilated
            gtEroded = self.erode(gt, kernel)
            changedChannels[channelNameFstPart + '_EroGT'] = gtEroded
            tipsForSynthetic = random.sample(tipsKeys, 4)
            i = 0
            for tip in tipsForSynthetic:
                scale = random.uniform(0.4, 1)
                tipScaled = tips[tip]['data'] * scale
                dilated = self.dilate(gt, tipScaled)
                eroded = self.erode(dilated, tipScaled)
                changedChannels[channelNameFstPart + '_Tip' + str(i)] = tipScaled
                changedChannels[channelNameFstPart + '_Dil' + str(i)] = dilated
                changedChannels[channelNameFstPart + '_Ero' + str(i)] = eroded
                i += 1

        self.addToPickle(pathToRealPickle, changedChannels)

    def loadBackgrounds(self):
        """
        Loads backgrounds from folder and normalizes them
        :return: fills backgrounds attribute
        """
        self.backgrounds = []
        fileNames = [f for f in os.listdir(self.pathToBackgrounds) if os.path.isfile(os.path.join(self.pathToBackgrounds, f))]
        for imagePath in fileNames:
            bckg = imageio.imread(os.path.join(self.pathToBackgrounds,imagePath))
            bckg = self.normalize(np.asarray(ShapeClass.resize(bckg, (self.resolution, self.resolution))))
            self.backgrounds.append(bckg)

    def generateData(self):
        """
        Generates ad saves data according to parameters give to the class in init. Len of test dataset is 1/5 of all data
        :return: fills specified directory with data
        """
        global paramsSEMSim
        if self.SEMSim:
            self.simulator = SEMSimulator(paramsSEMSim)

        self.loadBackgrounds()
        if not os.path.exists(os.path.join(self.dataFolder)):
            os.mkdir(os.path.join(self.dataFolder))
        if not os.path.exists(os.path.join(self.dataFolder, self.npyOrImages)):
            os.mkdir(os.path.join(self.dataFolder, self.npyOrImages))
            os.mkdir(os.path.join(self.dataFolder, self.npyOrImages, 'train'))
            os.mkdir(os.path.join(self.dataFolder, self.npyOrImages, 'test'))
            os.mkdir(os.path.join(self.dataFolder, self.npyOrImages, 'train', 'label'))
            os.mkdir(os.path.join(self.dataFolder, self.npyOrImages, 'test', 'label'))

        if self.dilation:
            tipsFile = open(self.tipsPath, 'rb')
            tips = pickle.load(tipsFile)
            tips = tips[list(tips.keys())[0]]
            tipsKeys = list(tips.keys())
        objects = self.createObjects()
        iterations = self.datasetLen // 3 if self.mode == 2 else self.datasetLen
        tipScaled = None

        for i in range(0, iterations):

            if self.generatorStep == 1 and  i != 0 and i // (iterations // 2) == 1:
                self.generatorStep = 2

            if self.dilation:
                tip = random.choice(tipsKeys)
                rot = np.random.randint(0, 3)
                tipScaled = self.normalize(tips[tip]['data'])
                scale = random.uniform(0.6, 1)
                tipScaled *= scale
                for j in range(0,rot):
                    tipScaled = np.rot90(tipScaled)
            angles = [random.randint(10, 60) for tmp in range(0, 3)]

            if random.uniform(0,1) < self.gridPercentage:
                sizesPercentage = (random.randint(25,40), random.randint(25,40))
                rot = random.randint(0,40)
                image = self.createGrid(self.resolution,(sizesPercentage),rot)
            else:
                image = self.randomImage(self.resolution, objects)
            self.saveNewImgs(tipScaled, image, angles)

        testIndices = random.sample(range(1, self.datasetLen), self.datasetLen // 5)
        appendix = '.jpg'
        if self.npyOrImages == 'npy':
            appendix = ''
        # for file_name in testIndices:
        #     shutil.move(os.path.join(self.dataFolder, self.npyOrImages, 'train', str(file_name) + str(appendix)),
        #                 os.path.join(self.dataFolder, self.npyOrImages, 'test'))
        #     shutil.move(os.path.join(self.dataFolder, self.npyOrImages, 'train', 'label', str(file_name) + str(appendix)),
        #                 os.path.join(self.dataFolder, self.npyOrImages, 'test', 'label'))
        #     if self.mode == 3:
        #         shutil.move(os.path.join(self.dataFolder, self.npyOrImages, 'train', 'label', str(file_name) + 'TIP' + str(appendix)),
        #                     os.path.join(self.dataFolder, self.npyOrImages, 'test', 'label'))

    def getPatches(self, size):
        """
        DEPRECATED
        :param size:
        :return:
        """
        counter = 0
        for i in range(1,14):
            afm = imageio.imread("dataProbeTrain/AFM" + str(i) + '.jpg')
            afm = self.normalize(afm)
            sem = imageio.imread("dataProbeTrain/SEM" + str(i) + '.jpg')
            sem = self.normalize(sem)
            for j in range(0,30):
                x = np.random.randint(0, np.shape(afm)[0] - size)
                y = np.random.randint(0, np.shape(afm)[1] - size)
                patchAFM = afm[x:x+size, y:y+size]
                patchSEM = sem[x:x+size, y:y+size]
                self.saveImg(patchAFM, "dataset/" + str(counter) + 'AFM.jpg')
                self.saveImg(patchSEM, "dataset/" + str(counter) + 'SEM.jpg')
                counter += 1

    def createStatsData(self):
        """
        DEPRECATED, for specific experiment
        :return:
        """
        bckg = imageio.imread("eval/bckgTestData.jpg")
        bckg = ShapeClass.normalize(bckg)
        angle = 7
        resolution = 512
        background = Rectangle((resolution, resolution))
        background.shapeArr = bckg
        background.rotateHeight(0.05, 0.05)
        background.resize((1024,1024))
        resolution = 1024
        objects = []
        flags = [8,9,10]
        for i in range(5,11):
            for j in range(0,2):
                deform = Deformation(os.path.join('deformations', 'deformation' + str(i) + '.jpg'), (200, 200))
                if random.uniform(0,1) < 0.5:
                    deform.createPyramid(7)
                objects.append(deform)

        objects = []
        deform = Deformation(os.path.join('deformations', 'deformation' + str(9) + '.jpg'), (200, 200))
        deform.createPyramid(4)
        objects.append(deform)
        for i in range(0,3):
            barrel2 = Barrel((100, 100))
            barrel2.invert()
            barrel2.mergeObjects(Barrel((50, 50)))
            objects.append(barrel2)
        #image = background.shapeArr
        image = np.zeros((1024,1024))
        a = 0
        for i in range(0, 20):
            index = random.randint(0, len(objects) - 1)
            maxSize = resolution//2
            shapeObj = objects[index]
            newSize = (random.randint(50, maxSize), random.randint(50, maxSize))
            newObj = copy.deepcopy(shapeObj)
            newObj.resize(newSize)
            newObj.shapeArr *= random.uniform(0.5,1)
            image = Synthesizer.addObject(newObj if newObj is not None else shapeObj, image, (random.randint(0, resolution), random.randint(0, resolution)), additive=True)
        image = self.blurImage(image, 1)

        image = ShapeClass.normalize(image)
        images = []
        for flag in flags:
            for f in range(1,5):
                img = Rectangle((1024, 1024))
                img.shapeArr = image
                img.rotate((f-1)*31)
                images.append(img.shapeArr)


                left = self.createArtifacts(images[-1], angle, direction=True, degreeSpread=0)
                right = self.createArtifacts(images[-1], angle, direction=False, degreeSpread=0)
                if flag == 8:
                    std = 0.0008
                    alpha = 0.99
                elif flag == 9:
                    std = 0.008
                    alpha = 0.95
                elif flag == 10:
                    std = 0.008
                    alpha = 0.8
                currentValL = 0.0
                currentValR = 0.0
                #p = (random.randint(256,256), random.randint(256,256))
                #left = left[p[0]:p[0]+512, p[1]:p[1]+512]
                #right = right[p[0]:p[0] + 512, p[1]:p[1] + 512]
                shape = np.shape(left)
                for i in range(0, shape[0]):
                    for j in range(0, shape[1]):
                        currentValL = alpha * currentValL + np.random.normal(0, std)
                        left[i, j] += currentValL
                        currentValR = alpha * currentValR + np.random.normal(0, std)
                        right[i, j] += currentValR
                print(left.min(), " ",left.max(), " ",right.min(), " ",right.max())
                left = self.normalize(left)
                right = self.normalize(right)
                if not i % 2:
                    dist = -0.12 if i == 2 else 0.12
                    ch = random.choice([True, False])
                    if ch:
                        left = left + dist
                        left[left > 1] = 1
                        left[left < 0] = 0
                    else:
                        right = right + dist
                        right[right > 1] = 1
                        right[right < 0] = 0
                #self.showImage(left)
                self.saveImg(left, os.path.join('eval', 'group' + str(flag), str(f) + 'L.png'))
                self.saveImg(right, os.path.join('eval', 'group' + str(flag), str(f) + 'R.png'))

paramsSyn = {
    "objNMin" : 10, # minimum number of objects added to synthetic image
    "objNMax" : 50, # maximum -||-
    "generatorStep" : 1, # initial state of generator
    "shiftProb" : 0.0, # probability of intensity shift between L and R images
    "shiftBound" : 0.1, # maximum possible intesity shift
    "noiseParams" : [0.001, 0.007, 0.8,0.999], # see function (equation) for noise generation
    "resolution" : 128, # images generated are square
    "pathToRealData" : "synthesizerData/newRealData.pkl",
    "tipsPath" : "synthesizerData/tips.pkl", # path to the tips used for dilation
    "npyOrImages" : "images", # "npy" or "images",
    "index" : 0, # initial index
    "dilation" : 1, # probability of dilation 0-1
    "gridPercentage": 0.2, # percentage of grids in generated data
    "datasetLen" : 30000,
    "dataFolder" : 'datasets', # where data should be generated
    "blur" : True, # should data be blurred
    "blurDataPercentage" : 0, # percentage of blurred data
    "mode" : 2, # mode 1 saves only dilated data 1 channel, mode 2 saves two channels (L,R), mode 3 saves also tip
    "gaussNoise" : False, # add gauss noise
    "noisePercentage" : 10, # percentage of noised data
    "pathToBackgrounds" : 'synthesizerData/backgrounds', # path to backgrouds used in some generated images
    "SEMSim" : False, # simulate also SEM?
    "bumps" : 0.1 # percentage of bumps added
}

syn = Synthesizer(paramsSyn)
syn.generateData()

# syn.createStatsData()
#ss = SEMSimulator(paramsSEMSim)
# start = random.uniform(0,math.pi * 2)
# diff = random.uniform(0, math.pi * 1.5)
# showImage(ShapeClass.normalize(imageio.imread(os.path.join("nenovisionTasks", "scalesMC2.jpg"))))
#img = ss.mcMethod3D(ShapeClass.normalize(imageio.imread(os.path.join("nenovisionTasks", "scalesAFM2.jpg"))), (0, math.pi))
#showImage(img)
