import cv2 as cv
import os
import numpy as np

root = os.getcwd()
img = cv.imread(os.path.join(root, 'lab03/UnityHall.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
surf = cv.xfeatures2d.SURF_create(5000)
surf.setUpright(True)
print( surf.descriptorSize() )
surf.setExtended(True)
kp, des = surf.detectAndCompute(gray,None)

img2=cv.drawKeypoints(gray,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imshow('SURF',img2)
cv.waitKey(0)
cv.destroyAllWindows()

