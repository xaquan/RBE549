import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

imgl = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'left.jpg'), cv.IMREAD_GRAYSCALE)
imgr = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'right.jpg'), cv.IMREAD_GRAYSCALE)

# resize images to 480
imgl = cv.resize(imgl, (480, 480))
imgr = cv.resize(imgr, (480, 480))

#rotate images 90 degrees clockwise
imgl = cv.rotate(imgl, cv.ROTATE_90_CLOCKWISE)
imgr = cv.rotate(imgr, cv.ROTATE_90_CLOCKWISE)

sift = cv.SIFT_create()

kpl, desl = sift.detectAndCompute(imgl, None)
kpr, desr = sift.detectAndCompute(imgr, None)

# draw key points
imgl_keypoints = cv.drawKeypoints(imgl, kpl, None, color=(0, 0, 255), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
imgr_keypoints = cv.drawKeypoints(imgr, kpr, None, color=(0, 255, 255), flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

# rotate keypoints images 90 degrees counterclockwise
imgl_keypoints = cv.rotate(imgl_keypoints, cv.ROTATE_90_COUNTERCLOCKWISE)
imgr_keypoints = cv.rotate(imgr_keypoints, cv.ROTATE_90_COUNTERCLOCKWISE)

# find matches using FLANN-based matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desl, desr, k=2)

ptsl = []
ptsr = []

# create a mask to filter out good matches
matchesMask = [[0,0] for i in range(len(matches))]
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
        ptsl.append(kpl[m.queryIdx].pt)
        ptsr.append(kpr[m.trainIdx].pt)

ptsl = np.int32(ptsl)
ptsr = np.int32(ptsr)

Flc, masklc = cv.findFundamentalMat(ptsl, ptsr, cv.FM_8POINT)



# draw matches using knnMatch, stack vertically the two images with keypoints and matches
draw_params = dict(matchColor = (255,255,255),
                   singlePointColor = (255,255,255),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

matches_img = cv.drawMatchesKnn(imgl, kpl, imgr, kpr, matches, None, **draw_params)
matches_img = cv.rotate(matches_img, cv.ROTATE_90_COUNTERCLOCKWISE)

keypoints_img = np.vstack((imgl_keypoints, imgr_keypoints))


cv.imshow('SIFT Keypoints', keypoints_img)
cv.imshow('SIFT Matches', matches_img)
cv.waitKey(0)