"""
Did not have time to optimize this but it should estimate the tip used for scanning ->
Algorithm from Villarubia but gwyddion implementation in C is MUCH faster.
I advise to install pygwy if u want to autamatically estimate tips. This alg
should be included in pygwy api or do it manually.

"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

class label(Exception):
        def __init__(self):
            pass

def estimatePoint(image, size, thresh, pixelCoords, tip0):
    ixp, jxp = pixelCoords
    tip_xsiz, tip_ysiz = size
    im_xsiz, im_ysiz = np.shape(image)
    xc, yc = (size[0] //2, size[1] // 2)
    interior = True#jxp>=tip_ysiz-1 and jxp<=im_ysiz-tip_ysiz and ixp>=tip_xsiz-1 and ixp<=im_xsiz-tip_xsiz
    count = 0
    inside = 1
    outside = 0
    if interior:
        for jx in range(0, tip_ysiz):
            for ix in range(0, tip_xsiz):
                imagep = image[ixp, jxp]
                dil = -1000
                for jd in range(0, tip_ysiz):
                    for id in range(0, tip_xsiz):
                        if (imagep - image[ixp + xc - id,jxp + yc - jd] > tip0[id,jd]):
                            continue
                        temp = image[ix + ixp - id][jx + jxp - jd] + tip0[id][jd] - imagep
                        dil = max(dil, temp)
                if (dil == -1000):
                    continue
                if dil < tip0[ix,jx] - thresh:
                    count += 1
                tip0[ix,jx] = dil + thresh if dil < tip0[ix,jx] - thresh else tip0[ix,jx]
        return count, tip0
    else:
        for jx in range(0, tip_ysiz):
            for ix in range(0, tip_xsiz):
                imagep = image[ixp, jxp]
                dil = -1000
                try:
                    for jd in range(0, tip_ysiz):
                        for id in range(0, tip_xsiz):
                            apexstate = outside
                            if (jxp+yc-jd < 0 or jxp+yc-jd >= im_ysiz or ixp+xc-id < 0 or ixp+xc - id >= im_xsiz):
                                apexstate = 1
                            elif (imagep-image[ixp + xc - id, jxp+yc-jd] < tip0[id][jd]):
                                apexstate = 1
                            if (jxp+jx-jd < 0 or jxp+jx-jd >= im_ysiz or ixp+ix-id < 0 or ixp+ix-id >= im_xsiz):
                                xstate = outside
                            else:
                                xstate = inside
                            if (apexstate == outside):
                                continue
                            if (xstate == outside):
                                    raise label()
                                    pass
                            temp = image[ix + ixp - id, jx + jxp - jd] + tip0[id,jd] - imagep
                            dil = max(dil, temp)
                    if dil == -1000:
                        continue
                    tip0[ix,jx] = dil + thresh if dil < tip0[ix,jx] - thresh else tip0[ix,jx]
                except label:
                    pass
        return count, tip0

def tipIter(image, size, thresh, tip0):
    tip_xsiz, tip_ysiz = size
    im_xsiz, im_ysiz = np.shape(image)
    xc, yc = (size[0] // 2, size[1] // 2)
    opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, tip0)
    count = 0
    for jxp in range(tip_ysiz - 1 - yc, im_ysiz - tip_ysiz):
        for ixp in range(tip_xsiz - 1 - xc, im_xsiz - tip_xsiz):
            if (image[ixp, jxp] - opening[ixp, jxp] > thresh):
                flag, tip0 = estimatePoint(image, size, thresh, (ixp, jxp), tip0)
                if flag:
                    count += 1
    return count, tip0

def showImage(data):
    fig = plt.figure()
    imgplot = plt.imshow(data, interpolation='none', origin='upper')
    plt.show()

def tipEstimate(image, size, thresh):
    tip0 = np.full(size, 0.1)
    count = 1
    iteration = 0
    while count:
        count, tip0 = tipIter(image, size, thresh, tip0)
        showImage(tip0)
        iteration += 1
        print("Finished iteration: ", iteration)