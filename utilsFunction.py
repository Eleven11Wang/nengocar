
import os
import cv2
import time
import nengo
import random
import nengo_dl
import numpy as np
#import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import rescale, resize, downscale_local_mean


def read_img(name, h, w):
    path = name
    img = plt.imread(path)
    # img = irgb2gray(img)

    img = resize(img, (h, w))
    img = img * 255
    img = img.astype(int)
    # img = img.astype('uint8')

    return img

def RGB2YUV(rgb):
    m = np.array([[0.29900, -0.16874, 0.50000],
                  [0.58700, -0.33126, -0.41869],
                  [0.11400, 0.50000, -0.08131]])

    yuv = np.dot(rgb, m)
    yuv[:, :, 1:] += 128.0
    return yuv


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray


def morphology(img):
    open_element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opend = cv2.morphologyEx(img, cv2.MORPH_OPEN, open_element)
    # 腐蚀
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(opend, kernel, iterations=3)
    return erosion



def add_rectangle(img,localization_ls):

    fig, ax = plt.subplots()
    h,w = img.shape
    plt.imshow(img)
    for pos in localization_ls:
        ax.add_patch(
                patches.Rectangle(
                    (pos[0]* w-14, pos[1]*h-14),
                    28 ,
                    28 ,
                    edgecolor='blue',
                    facecolor='None',
                    fill=False
                ))
    plt.show(cmap ="gray")


def to_binary(rtx):
    rt_val =[0,0]
    if rtx < 0.4:
        rt_val[0]= 1
    elif rtx > 0.6:
        rt_val[1]=1
    return rt_val
