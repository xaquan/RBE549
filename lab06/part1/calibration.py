import numpy as np
import cv2 as cv
import glob
import os

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIRECTORY, 'output')
IMAGE_DIR = os.path.join(CURRENT_DIRECTORY, 'images')

SAMPLE_IMAGE_BASE = "left12.jpg"

# write calibration data to a NumPy format file (npy) file in numpy format
def write_calibration_data(calibration_data):
    path = os.path.join(OUTPUT_DIR, "calibration_data.npz")
    print(f"Saving calibration data to {path}")
    np.savez(path, **calibration_data)        

def load_calibration_data():
    path = os.path.join(OUTPUT_DIR, "calibration_data.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Calibration data file '{path}' not found.")
    print(f"Loading calibration data from {path}")
    return np.load(path, allow_pickle=True)   

os.makedirs(OUTPUT_DIR, exist_ok=True)

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

image_size = []
calibration_data = {}
corner_images_paths = []  # Store corner sample for visualization

images = glob.glob(IMAGE_DIR + '/*.jpg')
for fname in images:
    img = cv.imread(fname)
    if img is None:
        continue
    print(f"Processing image: {fname}")
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)        

        # Draw and display the corners
        cv.drawChessboardCorners(img, (9,6), corners2, True)
        base = os.path.splitext(os.path.basename(fname))[0]
        path = os.path.join(OUTPUT_DIR, f"corners_{base}.jpg")
        cv.imwrite(path, img)
        corner_images_paths.append(path)

# print corner images paths
print("Corner images saved to:")
for path in corner_images_paths:
    print(path)

if len(objpoints) < 3:
    raise ValueError("Not enough valid images for calibration. At least 3 are required.")

# Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)
sample_img = cv.imread(os.path.join(IMAGE_DIR, SAMPLE_IMAGE_BASE))
h, w = sample_img.shape[:2]
gray = cv.cvtColor(sample_img, cv.COLOR_BGR2GRAY)

# Undistortion
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# print("Camera matrix:\n", mtx)
# print("Distortion coefficients:\n", dist)
calibration_data['Calibration matrix'] = mtx
calibration_data['Distortion coefficients'] = dist

# Undistort
undistorted_image = cv.undistort(sample_img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
undistorted_image = undistorted_image[y:y+h, x:x+w]

print("Save calibrated image to 'calibratedImage.png'")
cv.imwrite(os.path.join(OUTPUT_DIR, 'calibratedImage.png'), undistorted_image)

# re-projection error
mean_error = 0
for i in range(len(objpoints)):
    projected, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i].reshape(-1, 2), projected.reshape(-1, 2), cv.NORM_L2) / len(projected)
    mean_error += error

# print("Mean re-projection error:", mean_error/len(objpoints))
calibration_data['Mean re-projection error'] = mean_error/len(objpoints)

write_calibration_data(calibration_data)

calibration_data_loaded = load_calibration_data()
print("Loaded calibration data:")
for key, value in calibration_data_loaded.items():
    print(f"{key}:\n{value}\n")

# show corners and undistorted image
corner_image = cv.imread(os.path.join(OUTPUT_DIR, f"corners_{SAMPLE_IMAGE_BASE}"))

if corner_image.shape[0] != undistorted_image.shape[0]:
    scale = undistorted_image.shape[0] / corner_image.shape[0]
    corner_image = cv.resize(corner_image, (int(corner_image.shape[1] * scale), undistorted_image.shape[0]))
combined_image = np.hstack((corner_image, undistorted_image))

cv.imshow('Combined Image', combined_image)
key = cv.waitKey(0) & 0xFF
if key == ord('q'):
    pass

cv.destroyAllWindows()

