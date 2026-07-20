import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
imgL = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'aloe', 'aloe_l.png'), cv.IMREAD_GRAYSCALE)
imgR = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'aloe', 'aloe_r.png'), cv.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
    print("Error: Could not load images.")
    exit()

stereo = cv.StereoBM.create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL, imgR)

os.makedirs(os.path.join(CURRENT_DIRECTORY, 'output'), exist_ok=True)

filename = os.path.join(CURRENT_DIRECTORY, 'output', "disparity.jpg")
cv.imwrite(filename, disparity)

plt.imshow(disparity, 'gray')
plt.show()