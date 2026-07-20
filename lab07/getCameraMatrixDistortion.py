import glob

import cv2 as cv
import numpy as np
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []
# print(os.path.join(CURRENT_DIRECTORY, 'images', '*.jpeg'))
images = glob.glob(os.path.join(CURRENT_DIRECTORY, 'images', 'chessboard', '*.jpeg'))
if not images:
    print("Error: No images found in the 'images' directory.")
    exit()

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # cv.drawChessboardCorners(img, (9,6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
    
ret, mtx, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez(os.path.join(CURRENT_DIRECTORY, 'camera_calibration_data.npz'), mtx=mtx, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
data = np.load(os.path.join(CURRENT_DIRECTORY, 'camera_calibration_data.npz'))
mtx, dist_coeffs = data['mtx'], data['dist_coeffs']
print("Camera matrix:\n", mtx)
print("Distortion coefficients:\n", dist_coeffs)
# print("rvecs:\n", data['rvecs'])
# print("tvecs:\n", data['tvecs'])