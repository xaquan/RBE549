from fileinput import filename
import cv2 as cv
import os
from datetime import datetime
import numpy as np
from lib.panoramaSticher import PanoramaStitcher
import threading
import time
from time import sleep
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Define constants for the save folder path, zoom button properties, trackbar padding, and datetime position
SAVE_FOLDER_PATH = "saved_media"    
ZOOM_BTN_WIDTH = 20
ZOOM_BTN_COLOR = (200, 200, 200)  # gray color for buttons
ZOOM_BTN_TXT_COLOR = (0, 0, 0)  # black color for text
TRACKBAR_PADDING_LR = 80 
TRACKBAR_PADDING_BOTTOM = 50
THRESHOLD_VALUE = 180
WINDOW_NAME = "xCamera 3.0"

EXTRACT_COLOR_LOWER = np.array([150, 40, 60])  # lower bound for pink color in HSV
EXTRACT_COLOR_UPPER = np.array([320, 255, 255])  # upper bound

DATETIME_POS = (470, 470)

OPENCV_IMG_PATH = os.path.join(CURRENT_DIRECTORY, "openCV.png")
SEARCH_IMG_PATH = os.path.join(CURRENT_DIRECTORY, "device.jpg")

MIN_GOOD = 10

is_recording = False
is_extracting_color = False
is_thresholding = False
is_gaussian_blurring = False
is_sharpening = False
is_object_detection = False
sobel_wait_selecting = False
is_sobel_filter_X = False
is_sobel_filter_Y = False
is_panorama_mode = False
is_recording_hdr = False
sobel_kernel_size = (0,0)
is_canny_edge_detection = False
canny_threshold_1 = 0
canny_threshold_2 = 0
rotate_angle = 0
video_filename = None
video_out = None
zoom_factor = 1.0
zoom_range = (0.5, 1.0)  # zoom factor range from 1 to 0.5 or 1 to 5 (5x zoom in to no zoom)
guassian_blur_sigma = (5, 5)  # default sigma values for Gaussian blur
sift = None
search_img = None
nFeatures = 0
cache_images = []
manual_exposure_value = None  # Initialize manual_exposure_value to None


btn_zoom_out_pos = None
btn_zoom_in_pos = None


# function check if save folder exists, if not create it and save the frame in it
def create_save_folder():
    path = os.path.join(CURRENT_DIRECTORY, SAVE_FOLDER_PATH)
    if not os.path.exists(path):
        os.makedirs(path)
    print("Save folder created")    

# generate a file path for the photo or video to be saved in the save folder with the name "frame.jpg" or "video.avi"
def generate_file_path_with_timestamp(prefix, extension):
    path = os.path.join(CURRENT_DIRECTORY, SAVE_FOLDER_PATH)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{path}/{prefix}_{timestamp}.{extension}"

# add date and time to the frame at the bottom-right corner of the frame
def add_date_time_to_frame(frame):
    font = cv.FONT_HERSHEY_SIMPLEX
    datetime_stamp = datetime.today().strftime("%Y-%m-%d %H:%M")
    cv.putText(frame, datetime_stamp, DATETIME_POS, font, 0.5, (255, 255, 255), 1, cv.LINE_AA)

# save the current frame as a photo in the save folder with the name "img_<timestamp>.jpg"
def save_photo(frame):    
    filename = generate_file_path_with_timestamp("img", "jpg")
    create_save_folder()
    cv.imwrite(filename, frame)
    print(f"Frame captured and saved as '{filename}'")

# start recording video, create a VideoWriter object and save the video in the save 
# folder with the name "video.avi"
def start_recording(filename, frame_size, fps=10, exposure_value=None):
    global video_out
    create_save_folder()

    video_out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

# stop recording video, release the VideoWriter object and save the video in the 
# save folder with the name "video.avi"
def stop_recording():
    global video_out
    if video_out is not None:
        video_out.release()
        video_out = None

# add a red border to the frame
def add_border_to_frame(frame):
    return cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 255))


# change all pixels in the frame to white for 100 milliseconds to indicate that a photo has been taken
def flash_screen(frame):
    frame[:] = (255, 255, 255)
    cv.waitKey(100)  # wait for 100 milliseconds

# Show trackbars to control the sigma values for Gaussian blur, if show is True, create the trackbars, 
# if show is False, remove the trackbars by setting their positions to 0 and max value to 0
def show_gaussian_blur_trackbar(show=True):
    # Create a Adjust the trackbar to set the sigma values (X and Y) from (5-30)
    # window_name = "Gaussian filter"
    window_name = WINDOW_NAME
    if show:
        cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
        cv.createTrackbar("Sigma X", window_name, 0, 100, update_gaussian_x_blur)
        cv.createTrackbar("Sigma Y", window_name, 0, 100, update_gaussian_y_blur)
    else:
        # Remove the trackbars by setting their positions to 0 and max value to 0
        global guassian_blur_sigma
        cv.setTrackbarPos("Sigma X", window_name, 0)
        cv.setTrackbarPos("Sigma Y", window_name, 0)
        guassian_blur_sigma = (0, 0)
        # cv.destroyWindow(window_name)

# Update the global variable guassian_blur_sigma with the new sigma value for X or Y when the trackbar is 
# changed and print the new sigma values
def update_gaussian_x_blur(value):
    global guassian_blur_sigma
    normalized_sigma_x = 5 + (value / 100) * 25
    guassian_blur_sigma = (normalized_sigma_x, guassian_blur_sigma[1])
    print(f"Sigma: {guassian_blur_sigma}")

def update_gaussian_y_blur(value):
    global guassian_blur_sigma
    normalized_sigma_y = 5 + (value / 100) * 25
    guassian_blur_sigma = (guassian_blur_sigma[0], normalized_sigma_y)
    print(f"Sigma: {guassian_blur_sigma}")

# Apply Gaussian blur to the frame using the current sigma values in guassian_blur_sigma
def apply_gaussian_blur(frame, sigma):
    frame[:] = cv.GaussianBlur(frame, (0, 0), sigma[0], frame, sigma[1])

# add trackbars to control the kernel size for Sobel filter in X and Y direction, if type is "X", 
# create a trackbar for X direction, if type is "Y", create a trackbar for Y direction
def show_sobel_X_slider():
    global is_sobel_filter_X
    show_sobel_slider("X")
    is_sobel_filter_X = True

def show_sobel_Y_slider():
    global is_sobel_filter_Y
    show_sobel_slider("Y")
    is_sobel_filter_Y = True

def show_sobel_slider(type):
    # title = f"Sobel {type} filter"
    title = WINDOW_NAME
    callback = update_sobel_X_filter if type == "X" else update_sobel_Y_filter
    trackbar_name = f"Sobel {type}"
    cv.namedWindow(title, cv.WINDOW_AUTOSIZE)
    cv.createTrackbar(trackbar_name, title, 0, 10, callback)

# Update the global variable sobel_kernel_size with the new kernel size for X or Y direction 
# when the trackbar is changed and print the new kernel sizes
def update_sobel_X_filter(value):
    global sobel_kernel_size
    normalized_kernel_size = value * 2 + 1  # Ensure kernel size is odd and at least 1
    sobel_kernel_size = (normalized_kernel_size, sobel_kernel_size[1])
    print(f"Sobel X Kernel Size: {sobel_kernel_size[0]}")

def update_sobel_Y_filter(value):
    global sobel_kernel_size
    normalized_kernel_size = value * 2 + 1  # Ensure kernel size is odd and at least 1
    sobel_kernel_size = (sobel_kernel_size[0], normalized_kernel_size)
    print(f"Sobel Y Kernel Size: {sobel_kernel_size[1]}")

# Apply Sobel filter to the frame in X and/or Y direction based on the current kernel sizes in sobel_kernel_size 
# and update the frame with the resulting image
def apply_sobel_filter(frame, kernel_size):
    if kernel_size[0] > 1:
        sobel_x = cv.Sobel(frame, cv.CV_64F, 1, 0, ksize=kernel_size[0])
        frame[:] = cv.convertScaleAbs(sobel_x)
    if kernel_size[1] > 1:
        sobel_y = cv.Sobel(frame, cv.CV_64F, 0, 1, ksize=kernel_size[1])
        frame[:] = cv.convertScaleAbs(sobel_y)

# add trackbar to control the zoom level of the camera, the zoom level should be between 
def add_zoom_trackbar(frame: cv.Mat):
    """
    Add a zoom control trackbar to the frame with "+" and "-" buttons on either side.
    The trackbar allows the user to adjust the zoom level of the camera by clicking on the "+" or "-" buttons.
    """
    global btn_zoom_out_pos, btn_zoom_in_pos
    
    frame_height, frame_width = frame.shape[:2]

    btn_zoom_out_pos = (TRACKBAR_PADDING_LR, frame_height - TRACKBAR_PADDING_BOTTOM)
    btn_zoom_in_pos = (frame_width - TRACKBAR_PADDING_LR, frame_height - TRACKBAR_PADDING_BOTTOM)
    btn_text_out_pos = (btn_zoom_out_pos[0] - 13, btn_zoom_out_pos[1] + 10)
    btn_text_in_pos = (btn_zoom_in_pos[0] - 13, btn_zoom_in_pos[1] + 10)
    trackbar_pos_x_range = (btn_zoom_out_pos[0], btn_zoom_in_pos[0])
  
    trackbar_pos_y = btn_zoom_out_pos[1]
    trackbar_circle_radius = 10
    track_bar_circle_x_range = (trackbar_pos_x_range[0] + 30, trackbar_pos_x_range[1] - 30)

    
    cv.rectangle(frame, (trackbar_pos_x_range[0], trackbar_pos_y - 10), (trackbar_pos_x_range[1], trackbar_pos_y + 10), ZOOM_BTN_COLOR, -1)
    cv.circle(frame,btn_zoom_out_pos, ZOOM_BTN_WIDTH, ZOOM_BTN_COLOR, -1)
    cv.putText(frame, "-", btn_text_out_pos, cv.FONT_HERSHEY_SIMPLEX, 1, ZOOM_BTN_TXT_COLOR, 2, cv.LINE_AA)
    cv.circle(frame,btn_zoom_in_pos, ZOOM_BTN_WIDTH, ZOOM_BTN_COLOR, -1)
    cv.putText(frame, "+", btn_text_in_pos, cv.FONT_HERSHEY_SIMPLEX, 1, ZOOM_BTN_TXT_COLOR, 2, cv.LINE_AA)
    normalized_zoom = normalize_zoom_factor(zoom_factor)
    circle_pointer = int(track_bar_circle_x_range[0] + normalized_zoom * (track_bar_circle_x_range[1] - track_bar_circle_x_range[0]))

    cv.circle(frame, (circle_pointer, trackbar_pos_y), trackbar_circle_radius, ZOOM_BTN_TXT_COLOR, -1)

# Check if the mouse click is inside the button circle area
def is_inside_button(x, y, button_pos):
    """
    Check if the mouse click at (x, y) is inside the circular button area centered at button_pos.
    """
    global ZOOM_BTN_WIDTH
    distance = ((x - button_pos[0]) ** 2 + (y - button_pos[1]) ** 2) ** 0.5
    return distance <= ZOOM_BTN_WIDTH

# Normalize the zoom factor to a range of 0.0 to 1.0 inverting it so that 0.5 (max zoom in) maps to 1.0 and 1.0 (no zoom) maps to 0.0
def normalize_zoom_factor(zoom_factor):
    # Normalize the zoom factor to a range of 0.0 to 1.0 inverting it so that 0.5 (max zoom in) maps to 1.0 and 1.0 (no zoom) maps to 0.0
    return (1.0 - (zoom_factor - zoom_range[0]) / (zoom_range[1] - zoom_range[0]))

# Copy the region of interest (ROI) containing the date and time from the output frame to the show frame
# at the top-right corner of the frame
def copy_datetime_roi(output_frame, show_frame):

    frame_size = output_frame.shape[1], output_frame.shape[0]

    roi_width, roi_height = 180, 30

    copy_pos_x = frame_size[0] - roi_width
    copy_pos_y = frame_size[1] - roi_height

    paste_pos_x = frame_size[0] - roi_width
    paste_pos_y = 0

    copy_frame = output_frame[copy_pos_y :copy_pos_y + roi_height, copy_pos_x:copy_pos_x + roi_width]

    show_frame[paste_pos_y:paste_pos_y + roi_height, paste_pos_x:paste_pos_x + roi_width] = copy_frame

# Mouse callback function to handle zoom in and zoom out when the "+" or "-" buttons are clicked
def mouse_callback(event, x, y, flags, param):
    global button_clicked
    global zoom_factor

    if event == cv.EVENT_LBUTTONDOWN:
        # "+" should zoom in (crop smaller area), "-" should zoom out
        if is_inside_button(x, y, btn_zoom_out_pos):
            # zoom out: increase scale toward 1.0
            zoom_factor = min(zoom_range[1], zoom_factor + 0.1)
            print(f"Zoom out! Factor: {zoom_factor:.2f}")
        
        if is_inside_button(x, y, btn_zoom_in_pos):
            # zoom in: decrease scale but avoid zero
            zoom_factor = max(zoom_range[0], zoom_factor - 0.1)
            print(f"Zoom in! Factor: {zoom_factor:.2f}")

# Apply zoom to the frame based on the current zoom factor, crop the center of the frame and resize it back to the original size
def apply_zoom(frame, zoom_factor):
    if zoom_factor == 1.0:
        return

    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    x1 = max(0, center_x - new_width // 2)
    y1 = max(0, center_y - new_height // 2)
    x2 = min(width, center_x + new_width // 2)
    y2 = min(height, center_y + new_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]
    frame[:] = cv.resize(zoomed_frame, (width, height), interpolation=cv.INTER_LINEAR)

# Load the OpenCV image from the specified path and return it as a cv.Mat object, if the image does not exist print an error message and return None
def load_opencv_image():
    if os.path.exists(OPENCV_IMG_PATH):
        img = cv.imread(OPENCV_IMG_PATH)
        width = 100
        scale = width / img.shape[1]  # scale the image to a width of 200 pixels while maintaining aspect ratio
        newsize = (width, int(img.shape[0] * scale))
        img = cv.resize(img, newsize)
        return img
    else:
        print(f"OpenCV image not found at '{OPENCV_IMG_PATH}'. Please make sure the image exists.")
        return None

# Add the OpenCV image to the top-left corner of the frame if it exists
def add_opencv_image_to_frame(frame):
    opencv_img = load_opencv_image()
    if opencv_img is not None:
        img_height, img_width = opencv_img.shape[:2]
        frame[0:img_height, 0:img_width] = cv.addWeighted(opencv_img, 0.5, frame[0:img_height, 0:img_width], 0.5, 0)  # Blend the OpenCV image with the frame
        
# Extract the specified color from the frame using the provided lower and upper HSV bounds, 
# and return the resulting frame with only the extracted color visible
def extract_color(frame, color_lower, color_upper) -> cv.Mat:
    frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.inRange(frame_hsv, color_lower, color_upper)
    frame[:] = cv.bitwise_and(frame, frame, mask=mask)  # Apply the mask to the original frame to keep only the extracted color

# Rotate the frame by the specified angle and return the rotated frame
def rotate_frame(frame, angle):
    height, width = frame.shape[:2]
    center = (width // 2, height // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    frame[:] = cv.warpAffine(frame, M, (width, height))

# Threshold the frame with the specified threshold value, convert the frame to grayscale, apply the threshold, 
# and convert it back to BGR color space for consistency with other operations
def threshold_frame(frame, threshold_value):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, thresholded_frame = cv.threshold(gray_frame, threshold_value, 255, cv.THRESH_BINARY)
    frame[:] = cv.cvtColor(thresholded_frame, cv.COLOR_GRAY2BGR)  # Convert back to BGR for consistency with other operations
    # cv.imshow("Thresholded Frame", thresholded_frame)  # Display the thresholded frame for debugging
    # return thresholded_frame

# Sharpen the frame using a simple unsharp masking technique where the blurred image is subtracted 
# from the original image to get the detail, and then the detail is added back to the original image to get the sharpened image
def sharpen_frame(frame):
    copy = frame.copy()
    smoothed = cv.GaussianBlur(copy, (0, 0), 10)  # Apply Gaussian blur to the frame
    detail = cv.subtract(copy, smoothed)  # Get the detail by subtracting the smoothed image from the original image
    sharpened = cv.addWeighted(copy, 1, detail, 0.5, 0)  # Combine the original image with the detail image
    frame[:] = sharpened  # Update the frame with the sharpened image

# Show trackbars to control the Canny edge detection thresholds, if show is True, create the trackbars
def show_canny_edge_trackbars(frame):
    cv.createTrackbar("Canny Threshold 1", WINDOW_NAME, 1, 5000, update_canny_threshold_1)
    cv.createTrackbar("Canny Threshold 2", WINDOW_NAME, 1, 5000, update_canny_threshold_2)

# Update the global variables canny_threshold_1 and canny_threshold_2 with the new threshold values for 
# Canny edge detection when the trackbars are changed and print the new threshold values
def update_canny_threshold_1(value):
    global canny_threshold_1
    canny_threshold_1 = value
    print(f"Canny Threshold 1: {canny_threshold_1}")

def update_canny_threshold_2(value):
    global canny_threshold_2
    canny_threshold_2 = value
    print(f"Canny Threshold 2: {canny_threshold_2}")

# Apply Canny edge detection to the frame using the current threshold values in canny_threshold_1 
# and canny_threshold_2 and update the frame with the resulting edge-detected image
def apply_canny_edge_detection(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_frame, canny_threshold_1, canny_threshold_2)
    frame[:] = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)  # Convert back to BGR for consistency with other operations

# Lab03 Part 5: Object Detection and Feature Matching

# Detect keypoints and compute descriptors using the specified detector (SIFT or SURF)
# and return them as a tuple
def detectAndCompute(detector, img):
    kp, des = detector.detectAndCompute(img,None)
    return kp, des

# Brute-force matcher for SIFT/SURF, using L2 norm and no cross-checking, and return the matches
def brute_force_matcher(desc1, desc2, k=2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    return bf.knnMatch(desc1, desc2, k)

def get_matches_kp2(kp1, kp2, matches):
    return [kp2[m.trainIdx] for m in matches]

# Deaw matches between two images using the provided keypoints and matches, and return the resulting image
def draw_matches(img1,kp1,img2, kp2, matches, ratio=0.7, matchColor=(0, 255, 0)):
    # matchesMask = [[0,0] for i in range(len(matches))]
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < ratio*n.distance:
    #         matchesMask[i]=[1,0]

    # draw_params = dict(matchColor = matchColor,
    #                    singlePointColor = (255,0,0),
    #                    matchesMask = matchesMask,
    #                    flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # kp2_matches = [kp2[m.trainIdx] for m in matches]

    # # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

    # img3 = cv.drawKeypoints(img2, kp2_matches, None, color=matchColor)
    good_kp = [kp2[m.trainIdx]
           for m, n in matches
           if m.distance < ratio * n.distance]
    img3 = cv.drawKeypoints(img2, good_kp, None, color=matchColor)

    return img3

def object_detection(frame):
     # resize the device image height equat to the frame height while maintaining the aspect ratio
    img1 = cv.cvtColor(search_img, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   

    sift_kp1, sift_des1 = detectAndCompute(sift, img1)
    sift_kp2, sift_des2 = detectAndCompute(sift, img2)
    sift_brute_matches = brute_force_matcher(sift_des1, sift_des2)

    scene, mask = locate_object(sift_kp1, sift_kp2, match_mask(sift_brute_matches), search_img, frame)

    return scene

def match_mask(knn_matches, ratio=0.75):
    """Lowe's ratio test. Guards against degenerate pairs (<2 neighbours)."""
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

# Estimate homography from good matches and draw the projected box around the detected object in the scene image, 
# return the resulting image and the mask of inliers used for homography estimation
def locate_object(kp1, kp2, good, match_mask, img_train):
    """Estimate homography from good matches and draw the projected box."""
    if len(good) < MIN_GOOD:
        print(f"    Not enough matches: {len(good)}/{MIN_GOOD}")
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
    if H is None:
        print("    Homography estimation failed.")
        return None, None

    h, w = match_mask.shape[:2]
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    projected = cv.perspectiveTransform(corners, H)

    scene = img_train.copy()
    scene = cv.polylines(scene, [np.int32(projected)], True, (0, 255, 0), 3, cv.LINE_AA)
    inliers = int(mask.sum())
    print(f"    Homography OK -- {inliers}/{len(good)} inliers")
    return scene, mask

def update_sift_parameters(value):
    global nFeatures, sift
    nFeatures = value
    sift = cv.SIFT_create(nFeatures=nFeatures)
    print(f"SIFT nFeatures: {nFeatures}")

def cache_image(image):
    global cache_images
    cache_images.append(image)

def save_panorama_photo():
    global cache_images
    panorama_stitcher = PanoramaStitcher()
    if len(cache_images) > 1:
        panorama_stitcher.addImages(cache_images)
        panorama = panorama_stitcher.exportPanorama()
        if panorama is not None:
            # cv.imshow("Panorama", panorama)
            save_photo(panorama)
        else:
            print("Panorama stitching failed.")
    cache_images = []

# Lab05 Part 2: Main function to capture video from the camera, apply various image processing techniques, 
# and handle user input for recording, capturing photos, and toggling features

def testing_thread(threadId):
    print(f"Testing thread {threadId} started...")
    while is_recording_hdr:
        print(f"Thread {threadId} is running...")
        sleep(2)  # Simulate some work being done in the thread
    print(f"Testing thread {threadId} finished.")

# Save 3 exposure video by taking the frames and saving them as a video file with the specified filename and fps
# Use threading to save the video in the background while the main thread continues to capture frames from the camera
def save_multi_exposure_video(frame, filename, fps=20):
    # create thread
    global is_recording_hdr
    is_recording_hdr = True
    thread1 = threading.Thread(target=testing_thread, args=(filename,))  # Pass a thread ID as an argument
    # thread2 = threading.Thread(target=testing_thread, args=(1,))  # Pass a thread ID as an argument

    thread1.start()
    # thread2.start()


def manual_exposure_callback(value):
    global manual_exposure_value
    manual_exposure_value = value

def main():
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WINDOW_NAME, mouse_callback)

    # initialize the camera
    cap = cv.VideoCapture(0)

    # set the video format to MJPG
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    

    if not cap.isOpened():
        print("Cannot open camera")
        exit() 

    while True:   
        global is_recording, video_filename, video_out, is_extracting_color, rotate_angle
        global is_thresholding, is_gaussian_blurring, guassian_blur_sigma, is_sharpening
        global sobel_wait_selecting, is_sobel_filter_X, is_sobel_filter_Y, sobel_kernel_size
        global is_canny_edge_detection, is_object_detection, sift, search_img, nFeatures
        global is_panorama_mode, cache_images
        global is_recording_hdr, manual_exposure_value

        sift_brute_img = None

        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Apply zoom to the frame based on the current zoom factor and display it
        if zoom_factor != 1.0:
            apply_zoom(frame, zoom_factor)
        
        # Create a copy of the show_frame to apply transformations without modifying the original frame that will be displayed
        if is_extracting_color:
            extract_color(frame, EXTRACT_COLOR_LOWER, EXTRACT_COLOR_UPPER)

        # rotate the frame by the specified angle if rotate_angle is not zero
        if rotate_angle != 0:
            rotate_frame(frame, rotate_angle)
    
        # threshold the frame with a threshold value of 128 if is_thresholding is True
        if is_thresholding:
            
            threshold_frame(frame, THRESHOLD_VALUE)

        # Apply Gaussian blur to the frame if is_gaussian_blurring is True using the current sigma values from the trackbars
        if is_gaussian_blurring and (guassian_blur_sigma[0] > 0 or guassian_blur_sigma[1] > 0):
            apply_gaussian_blur(frame, guassian_blur_sigma)

        # Sharpen the frame if is_sharpening is True using a simple unsharp masking technique 
        # where the blurred image is subtracted from the original image to get
        if is_sharpening:
            sharpen_frame(frame)

        # Apply Sobel filter to the frame if is_sobel_filter_X or is_sobel_filter_Y is 
        # True using the current kernel sizes from the trackbars
        if is_sobel_filter_X or is_sobel_filter_Y:
            apply_sobel_filter(frame, sobel_kernel_size)

        # Apply Canny edge detection to the frame if is_canny_edge_detection is True 
        # using the current threshold values from the trackbars
        if is_canny_edge_detection and (canny_threshold_1 > 0 or canny_threshold_2 > 0):
            apply_canny_edge_detection(frame)

        # Perform object detection and feature matching using SIFT 
        # if is_object_detection is True, and update the frame with the resulting image
        if is_object_detection:
           sift_brute_img = object_detection(frame)

        if is_recording_hdr:                
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # 0.25 = manual on many UVC drivers (driver-dependent)
            cap.set(cv.CAP_PROP_EXPOSURE, manual_exposure_value)  # Set the exposure value to the current manual_exposure_value
        else:
            cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 3)  # Set the exposure value to 0 (auto)



        # Create a copy of the show_frame to add date and time without modifying the 
        # original frame that will be displayed.
        # If panorama mode is enabled, do not add date and time to the frame.
        output_frame = frame.copy()
        if not is_panorama_mode:
            add_date_time_to_frame(output_frame)

        key = cv.waitKey(1) & 0xFF
        
        if key == ord('v'):
            if not is_recording:
                is_recording = True
                video_filename = generate_file_path_with_timestamp("video", "mp4")
                start_recording(video_filename, (output_frame.shape[1], output_frame.shape[0]))
                
                print("Recording started...")
            else:
                is_recording = False
                stop_recording()                
                print(f"Recording stopped. Video saved as '{video_filename}'")
    
        elif key == ord('c'):
            flash_screen(frame)
            if is_panorama_mode:
                print("Capturing frame for panorama stitching...")
                cache_image(output_frame)
            else:
                save_photo(output_frame)
        
        # if pressed key is "e", extract pink
        elif key == ord('e'):
            is_extracting_color = not is_extracting_color

            if is_extracting_color:
                print("Extracting color...")
            else:
                print("Stop extracting color...")
            
        # if pressed key is "r", rotate by 10 degrees
        elif key == ord('r'):
            print("Rotating frame by 10 degrees...")
            if rotate_angle < 10:
                rotate_angle = 10
            else:
                rotate_angle = 0
            
        # if pressed key is "t", threshold the frame with a threshold value of 128
        elif key == ord('t'):
            is_thresholding = not is_thresholding
            if is_thresholding:
                print("Thresholding frame with value 128...")
            else:
                print("Stopping thresholding...")

        # if pressed key is "b", show trackbar to control gausian blur
        elif key == ord('b'):
            
            if not is_gaussian_blurring:                
                is_gaussian_blurring = True
                show_gaussian_blur_trackbar(True)
                print("Applying Gaussian blur to frame...")
            else:
                show_gaussian_blur_trackbar(False)
                print("Remove Gaussian blur...")
        
        # if pressed key is "s" extract sharpened image
        elif key == ord('s'):
            is_sharpening = not is_sharpening
            if is_sharpening:
                print("Extracting sharpened image...")
            else:
                print("Stopping sharpening...")

        elif key == ord('g') or sobel_wait_selecting:
            if not sobel_wait_selecting:
                print("Press \"x\" to apply Sobel filter in X direction...")
                print("Press \"y\" to apply Sobel filter in Y direction...")
                sobel_wait_selecting = True
            else:
                if key == ord('x'):
                    show_sobel_X_slider()
                    print("Show Sobel filter in X direction...")
                    sobel_wait_selecting = False
                    
                elif key == ord('y'):
                    show_sobel_Y_slider()
                    print("Show Sobel filter in Y direction...")
                    sobel_wait_selecting = False
        elif key == ord('d'):
            show_canny_edge_trackbars(frame)
            is_canny_edge_detection = True
            print("Show Canny edge detection trackbars...")
        
        elif key == ord('o'):         
            is_object_detection = not is_object_detection
            if is_object_detection:
                print("Object detection ...")                
                sift = cv.SIFT_create()
                search_img = cv.imread(SEARCH_IMG_PATH)                
                display_scale = frame.shape[0] / search_img.shape[0]
                search_img = cv.resize(search_img, (int(search_img.shape[1] * display_scale), frame.shape[0]))
                cv.imshow("Search Image", search_img)
            else:
                print("Stop object detection...")
                sift = None
                search_img = None
                cv.destroyWindow("Search Image")
        
        elif key == ord('h'):
            control_window_name = "Exposure Control"
            if not is_recording_hdr:                
                is_recording_hdr = True
                print("Recording multi-exposure video...")
                manual_exposure_value = int(cap.get(cv.CAP_PROP_EXPOSURE))  # Get the current exposure value from the camera
                cv.imshow(control_window_name, np.zeros((10, 400, 3), dtype=np.uint8))
                cv.createTrackbar("Exposure", control_window_name, manual_exposure_value, 1000, manual_exposure_callback)
                # save_multi_exposure_video(frame, "multi_exposure_video1.mp4")
            else:
                is_recording_hdr = False
                cv.destroyWindow(control_window_name)
                print("Stop recording multi-exposure video...")

        elif key == ord('p'):
            is_panorama_mode = not is_panorama_mode
            if is_panorama_mode:
                print("Panorama mode, capture multiple frames by pressing \"c\" and then stitch them together...")
            else:
                print("Panorama mode disabled.")
                save_panorama_photo()

        elif key == 27:  # ESC key to exit
            print("Exiting...")
            break

        
        # If recording is active, write the current frame to the video file 
        # and add a recording indicator to the frame
        if is_recording:
            if video_out is not None:
                video_out.write(output_frame)

        # # Copy the region of interest (ROI) containing the date and time from the output frame to the show frame at the top-right corner of the frame
        # copy_datetime_roi(output_frame, frame)

        # # Add the OpenCV image to the top-left corner of the frame if it exists
        # add_opencv_image_to_frame(frame)

        # # Add a red border to the frame to indicate that the camera is active
        # frame = add_border_to_frame(frame)        

        if is_object_detection and sift_brute_img is not None:
            frame = sift_brute_img

        # Add a zoom trackbar to the frame
        if not is_panorama_mode:
            add_zoom_trackbar(frame)

        # Display the resulting frame
        cv.imshow(WINDOW_NAME, frame)
    
    cap.release()
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()





