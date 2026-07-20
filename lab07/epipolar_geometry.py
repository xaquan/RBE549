import numpy as np
import cv2 as cv
import glob
import os
from matplotlib import pyplot as plt

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

imgc = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'globe', 'globe_center.jpg'), cv.IMREAD_GRAYSCALE)
imgl = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'globe', 'globe_left.jpg'), cv.IMREAD_GRAYSCALE)
imgr = cv.imread(os.path.join(CURRENT_DIRECTORY, 'images', 'globe', 'globe_right.jpg'), cv.IMREAD_GRAYSCALE)

if imgc is None or imgl is None or imgr is None:
    print("Error: Could not load images.")
    exit()

sift = cv.SIFT_create()
kpc, desc = sift.detectAndCompute(imgc, None)
kpl, desl = sift.detectAndCompute(imgl, None)
kpr, desr = sift.detectAndCompute(imgr, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)
matcheslc = flann.knnMatch(desl, desc, k=2)
matchescr = flann.knnMatch(desc, desr, k=2)

ptslc1 = []
ptslc2 = []
ptscr1 = []
ptscr2 = []

for i, (m, n) in enumerate(matcheslc):
    if m.distance < 0.97 * n.distance:
        ptslc2.append(kpc[m.trainIdx].pt)
        ptslc1.append(kpl[m.queryIdx].pt)

for i, (m, n) in enumerate(matchescr):
    if m.distance < 0.97 * n.distance:
        ptscr2.append(kpr[m.trainIdx].pt)
        ptscr1.append(kpc[m.queryIdx].pt)

ptslc1 = np.int32(ptslc1)
ptslc2 = np.int32(ptslc2)
Flc, masklc = cv.findFundamentalMat(ptslc1, ptslc2, cv.FM_LMEDS)

ptscr1 = np.int32(ptscr1)
ptscr2 = np.int32(ptscr2)
Fcr, maskcr = cv.findFundamentalMat(ptscr1, ptscr2, cv.FM_8POINT)

ptslc1 = ptslc1[masklc.ravel() == 1]
ptslc2 = ptslc2[masklc.ravel() == 1]
ptscr1 = ptscr1[maskcr.ravel() == 1]
ptscr2 = ptscr2[maskcr.ravel() == 1]

def drawlines(img1, img2, lines, pts1, pts2):
    r, c = img1.shape
    img1_color = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
    img2_color = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1_color = cv.line(img1_color, (x0, y0), (x1, y1), color, 1)
        img1_color = cv.circle(img1_color, tuple(pt1.astype(int)), 5, color, -1)
        img2_color = cv.circle(img2_color, tuple(pt2.astype(int)), 5, color, -1)
    return img1_color, img2_color

lineslc1 = cv.computeCorrespondEpilines(ptslc2.reshape(-1, 1, 2), 2, Flc)
lineslc1 = lineslc1.reshape(-1, 3)
imglc1, imglc2 = drawlines(imgl, imgc, lineslc1, ptslc1, ptslc2)

lineslc2 = cv.computeCorrespondEpilines(ptslc1.reshape(-1, 1, 2), 1, Flc)
lineslc2 = lineslc2.reshape(-1, 3)
imglc3, imglc4 = drawlines(imgc, imgl, lineslc2, ptslc2, ptslc1)

linescr1 = cv.computeCorrespondEpilines(ptscr2.reshape(-1, 1, 2), 2, Fcr)
linescr1 = linescr1.reshape(-1, 3)
imgcr1, imgcr2 = drawlines(imgc, imgr, linescr1, ptscr1, ptscr2)

linescr2 = cv.computeCorrespondEpilines(ptscr1.reshape(-1, 1, 2), 1, Fcr)
linescr2 = linescr2.reshape(-1, 3)
imgcr3, imgcr4 = drawlines(imgr, imgc, linescr2, ptscr2, ptscr1)


fig = plt.figure(figsize=(8, 8), )
plt.subplot(221), plt.imshow(imglc1, aspect='equal'), plt.title('Left Image with Epilines'), plt.axis('off')
plt.subplot(222), plt.imshow(imglc3, aspect='equal'), plt.title('Center Image with Epilines'), plt.axis('off')

plt.subplot(223), plt.imshow(imgcr1, aspect='equal'), plt.title('Center Image with Epilines'), plt.axis('off')
plt.subplot(224), plt.imshow(imgcr3, aspect='equal'), plt.title('Right Image with Epilines'), plt.axis('off')  
plt.tight_layout()
plt.show()
