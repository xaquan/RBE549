import os
import numpy as np
import cv2 as cv

root = os.getcwd()
img = cv.imread(os.path.join(root, 'lab03/UnityHall.png'))
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create(19000)
kp = sift.detect(gray,None)

print(f'Number of keypoints detected: {len(kp)}')

img=cv.drawKeypoints(gray,kp,img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('SIFT',img)
cv.waitKey(0)
cv.destroyAllWindows()