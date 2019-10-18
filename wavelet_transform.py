import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pywt
import os

def DWT(img):
    coeffs2 = pywt.dwt2(img, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2     #500x402x4

    print(LL.shape)
    print(LH.shape)
    titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

def test():
    # Load image
    original = pywt.data.camera()
    print(original.shape)
    cv2.imshow("ori", original)
    cv2.waitKey(0)
    # Wavelet transform of image, and plot approximation and details
    titles = ['Approximation', ' Horizontal detail',
            'Vertical detail', 'Diagonal detail']
    coeffs2 = pywt.dwt2(original, 'bior1.3')
    LL, (LH, HL, HH) = coeffs2
    fig = plt.figure(figsize=(12, 3))
    for i, a in enumerate([LL, LH, HL, HH]):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    plt.show()

DWT(cv2.imread("b1.jpg", 0))
# test()