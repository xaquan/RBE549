import cv2 as cv
import numpy as np
import os
import glob

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIRECTORY, 'output')
IMAGE_DIR = os.path.join(CURRENT_DIRECTORY, 'images')
UNDISTORTED_IMAGE_DIR = os.path.join(CURRENT_DIRECTORY, 'undistorted_images')
DISTORTED_IMAGE_DIR = os.path.join(CURRENT_DIRECTORY, 'distorted_images')
COMPARISON_IMAGE_DIR = os.path.join(CURRENT_DIRECTORY, 'comparison_images')
CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
CHECKERBOARD_DETECT_SIZE = (9, 6)
SQUARE_SIZE = 1

DISTORTION_COEFFICIENTS = np.array([[0.4, 0.08, 0.001, 0.001, -0.02]])

distorted_data = None

estimate_distortion_data = {}

with np.load(os.path.join(CURRENT_DIRECTORY, 'camera_calibration_data.npz')) as data:
    mtx, dist_coeffs = data['mtx'], data['dist_coeffs']

# Create ground truth camera matrix based on the image size
def create_ground_truth_camera_matrix(width, height):
    """
    Create a ground truth camera matrix based on the image width and height.
    """
    focal_length = max(width, height)  # Simple assumption for focal length
    center_x = width / 2
    center_y = height / 2
    camera_matrix = np.array([[focal_length, 0, center_x],
                               [0, focal_length, center_y],
                               [0, 0, 1]], dtype=np.float32)
    return camera_matrix

# Apply distortion to an image using the ground truth camera matrix and distortion coefficients
def apply_distort_image(image, K, dist_coeffs):
    """
    Distort an image using the ground truth camera matrix and distortion coefficients.
    """    
    global distorted_data
    if distorted_data is None:
        # print("Distorted data is not initialized. Initializing now.")
        distorted_data = {
            "camera_matrix": K,
            "distortion_coefficients": dist_coeffs
        }

    h, w = image.shape[:2]

    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 0, (w,h))
    distorted_image = cv.undistort(image, K, dist_coeffs, None, new_camera_matrix)
    return distorted_image


def apply_distort_images(image_dir, output_dir, dist_coeffs):
    """
    Distort all images in the specified directory and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(image_dir, "*.jpeg"))
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        image_size = img.shape[::-1][1:]  # Get (width, height)
        # K = create_ground_truth_camera_matrix(image_size[0], image_size[1])
        K = mtx  # Use the loaded camera matrix from calibration data
        distorted_img = apply_distort_image(img, K, dist_coeffs)
        base_name = os.path.basename(fname)
        output_path = os.path.join(output_dir, f"{base_name}")
        cv.imwrite(output_path, distorted_img)

def get_object_points_and_image_points(image_dir):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((CHECKERBOARD_DETECT_SIZE[1]*CHECKERBOARD_DETECT_SIZE[0],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHECKERBOARD_DETECT_SIZE[0],0:CHECKERBOARD_DETECT_SIZE[1]].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    image_size = None

    images = glob.glob(os.path.join(image_dir, "*.jpeg"))
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = gray.shape[::-1]

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD_DETECT_SIZE, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), CRITERIA)
            imgpoints.append(corners2)

    if len(objpoints) < 3:
        raise ValueError("Not enough valid images for calibration. At least 3 are required.")

    return objpoints, imgpoints, image_size

# Undistort an image using the given camera matrix and distortion coefficients
def undistort_image(image, K, dist_coeffs):
    """
    Undistort an image using the given camera matrix and distortion coefficients.
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(K, dist_coeffs, (w,h), 0, (w,h))
    undistorted_image = cv.undistort(image, K, dist_coeffs, None, new_camera_matrix)
    # x, y, w, h = roi
    # undistorted_image = undistorted_image[y:y+h, x:x+w]
    return undistorted_image

# Undistort all images in the specified directory and save them to the output directory.
def undistort_images(image_dir, output_dir, K, dist_coeffs):
    """
    Undistort all images in the specified directory and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    images = glob.glob(os.path.join(image_dir, "*.jpeg"))
    
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            continue
        undistorted_img = undistort_image(img, K, dist_coeffs)            
        base_name = os.path.basename(fname)
        output_path = os.path.join(output_dir, f"{base_name}")
        cv.imwrite(output_path, undistorted_img)

def create_comparison_image(original_image, distorted_image, undistorted_image):
    """
    Create a comparison image that shows the original, distorted, and undistorted images side by side.
    """
    # Resize images to the same height
    height = min(original_image.shape[0], distorted_image.shape[0], undistorted_image.shape[0])
    original_resized = cv.resize(original_image, (int(original_image.shape[1] * height / original_image.shape[0]), height))
    distorted_resized = cv.resize(distorted_image, (int(distorted_image.shape[1] * height / distorted_image.shape[0]), height))
    undistorted_resized = cv.resize(undistorted_image, (int(undistorted_image.shape[1] * height / undistorted_image.shape[0]), height))

    # Concatenate images horizontally
    comparison_image = np.vstack((original_resized, distorted_resized, undistorted_resized))
    return comparison_image

# Create comparison images for all images in the original directory and save them to the output directory.
def create_comparison_images(original_dir, distorted_dir, undistorted_dir, output_dir):
    """
    Create comparison images for all images in the original directory and save them to the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    original_images = glob.glob(os.path.join(original_dir, "*.jpeg"))
    for original_path in original_images:
        base_name = os.path.basename(original_path)
        distorted_path = os.path.join(distorted_dir, f"{base_name}")
        undistorted_path = os.path.join(undistorted_dir, f"{base_name}")

        if not os.path.exists(distorted_path) or not os.path.exists(undistorted_path):
            print(f"Skipping {base_name} as distorted or undistorted image is missing.")
            continue

        original_image = cv.imread(original_path)
        distorted_image = cv.imread(distorted_path)
        undistorted_image = cv.imread(undistorted_path)

        comparison_image = create_comparison_image(original_image, distorted_image, undistorted_image)
        output_path = os.path.join(output_dir, f"comparison_{base_name}")
        cv.imwrite(output_path, comparison_image)

# Compute reprojection error
def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv.norm(imgpoints[i].reshape(-1, 2), imgpoints2.reshape(-1, 2), cv.NORM_L2) / len(imgpoints2)
        total_error += error
    mean_error = total_error / len(objpoints)
    return mean_error

if __name__ == "__main__":
    
    # Distort images in the IMAGE_DIR and save them to DISTORTED_IMAGE_DIR
    apply_distort_images(IMAGE_DIR, DISTORTED_IMAGE_DIR, DISTORTION_COEFFICIENTS)

    # Get object points and image points from the distorted images
    objpoints, imgpoints, image_size = get_object_points_and_image_points(DISTORTED_IMAGE_DIR)

    ret, K_est, dist_est, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    undistort_images(DISTORTED_IMAGE_DIR, UNDISTORTED_IMAGE_DIR, K_est, dist_est)

    create_comparison_images(IMAGE_DIR, DISTORTED_IMAGE_DIR, UNDISTORTED_IMAGE_DIR, COMPARISON_IMAGE_DIR)

    # Compute reprojection error
    reprojection_error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K_est, dist_est)
    print(f"Mean reprojection error: {reprojection_error}")

    global estimated_distortion_data
    estimated_distortion_data = {
        "estimated_camera_matrix": K_est,
        "estimated_distortion_coefficients": dist_est,
        "mean_reprojection_error": reprojection_error
    }

    print("Estimated data:")
    for key, value in estimated_distortion_data.items():
        print(f"{key}:\n{value}\n")

    print("Distorted data:")
    for key, value in distorted_data.items():
        print(f"{key}:\n{value}\n")

    # cv.imshow("Combined Image", combined_image)
    # cv.waitKey(0)