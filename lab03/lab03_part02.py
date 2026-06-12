import cv2 as cv
import os
import numpy as np

root = os.getcwd()
img1 = cv.imread(os.path.join(root, 'lab03/device.jpg'), cv.IMREAD_GRAYSCALE)
img2 = cv.imread(os.path.join(root, 'lab03/device_in_scene.jpg'), cv.IMREAD_GRAYSCALE)

orb = cv.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)


bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv.drawMatches(img1,kp1,img2,kp2,matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv.imshow('ORB',img3)
cv.waitKey(0)
cv.destroyAllWindows()