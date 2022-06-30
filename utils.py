from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import argparse

import cv2
from glob import glob
from copy import deepcopy
from sklearn.mixture import GaussianMixture

def min_max_scale(img):
    mx = img.max()
    mi = img.min()
    return (img-mi)/(mx-mi)

def laplacian_sharper(img):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
    imgLaplacian = cv2.filter2D(img, cv2.CV_32F, kernel)
    
    return min_max_scale(-min_max_scale(imgLaplacian) + min_max_scale(img))

def preprocess(img, cvtTo):
    img = (laplacian_sharper(img)*255).astype('uint8')
    if cvtTo == 'RGB':
        return img
    if cvtTo == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if cvtTo == 'LAB':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if cvtTo == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

    return img

def plotImage(img):
    img2 = deepcopy(img)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_HSV)
    plt.imshow(img2)
    #plt.title('Clustered Image')
    plt.show()
    
def GMM_cluster(img):

    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    gmm  = GaussianMixture(2, init_params='kmeans').fit(image_2D)
    clustOut = gmm.means_[gmm.predict(image_2D)]

    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])

    clusteredImg = np.uint8(clustered_3D*255)

    return clusteredImg

def edgeDetection(clusteredImage):
    edged = cv2.Canny(clusteredImage, 0, 255)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def contours(edgedImg):
    contours, hierarchy = cv2.findContours(edgedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    rect = []
    rect_with_edges = []
    contours_smooth = []

    for i, c in enumerate(contours):
        contours_smooth.append(cv2.approxPolyDP(c, 3, True))
        tmp = cv2.minAreaRect(contours_smooth[i])
        rect.append(tmp)
        rect_with_edges.append(tmp)
        rect[i] = cv2.boxPoints(rect[i])
        rect[i] = np.int0(rect[i])
    return rect[0], rect_with_edges[0]

def l2(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
  
def h_w(rect):
    x = rect[0]
    h_w = []
    for point in rect[1:]:
        h_w.append(l2(x, point))
    h_w.sort()
    return np.array([h_w[0], h_w[1]])