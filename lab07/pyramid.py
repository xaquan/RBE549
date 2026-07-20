import numpy as np
import cv2 as cv
import glob
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

with np.load(os.path.join(CURRENT_DIRECTORY, 'camera_calibration_data.npz')) as data:
    mtx, dist = data['mtx'], data['dist_coeffs']

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    img = cv.circle(img, tuple(imgpts[3].ravel()), 5, (0,0,255), -1)
    return img

def draw_cube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

def draw_triangular(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    apex_i = 3
    img = cv.drawContours(img, [imgpts[:3]],-1,(0,255,0),-3)
    for i in range(3):        
        next_i = (i + 1) % 3
        contour = np.array([imgpts[i], imgpts[next_i], imgpts[apex_i]])
        img = cv.drawContours(img, [contour],-1,(0, 0, 255),3)

    # img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# axis = np.float32([[3,0,0], [0,3,0], [0,0, -3], [0, 0, 0]]).reshape(-1,3)

side_length = 3
corner3_x = 1.5
corner3_y = np.sqrt(side_length**2 - corner3_x**2)
center_x = (side_length + corner3_x) / 3
center_y = corner3_y / 3
center_h = np.sqrt(side_length**2 - center_x**2)

axis_triangular = np.float32([[0,0,0], [side_length,0,0], [side_length/2,corner3_y,0],
                   [center_x,center_y,-center_h] ])

for fname in glob.glob(os.path.join(CURRENT_DIRECTORY, 'images', 'chessboard', '*.jpeg')):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        ret,rvecs,tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        # axis_imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        triangular_imgpts, jac = cv.projectPoints(axis_triangular, rvecs, tvecs, mtx, dist)

        # img = draw(img, corners2, axis_imgpts)
        img = draw_triangular(img, corners2, triangular_imgpts)
        cv.imshow('img',img)
        k = cv.waitKey(0) & 0xFF
        if k == ord('s'):
            os.makedirs(os.path.join(CURRENT_DIRECTORY, 'output'), exist_ok=True)
            filename = os.path.join(CURRENT_DIRECTORY, 'output', 'triangular_' + os.path.basename(fname))
            cv.imwrite(filename, img)
        if k == 27:
            break

cv.destroyAllWindows()