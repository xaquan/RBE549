from fileinput import filename
import cv2 as cv
import os
from datetime import datetime
import numpy as np

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Define constants for the save folder path, zoom button properties, trackbar padding, and datetime position
SAVE_FOLDER_PATH = "saved_media"    
ZOOM_BTN_WIDTH = 20
ZOOM_BTN_COLOR = (200, 200, 200)  # gray color for buttons
ZOOM_BTN_TXT_COLOR = (0, 0, 0)  # black color for text
TRACKBAR_PADDING_LR = 80 
TRACKBAR_PADDING_BOTTOM = 50
THRESHOLD_VALUE = 180
WINDOW_NAME = "xCamera"

EXTRACT_COLOR_LOWER = np.array([150, 40, 60])  # lower bound for pink color in HSV
EXTRACT_COLOR_UPPER = np.array([320, 255, 255])  # upper bound

DATETIME_POS = (470, 470)

OPENCV_IMG_PATH = os.path.join(CURRENT_DIRECTORY, "openCV.png")


is_recording = False
is_extracting_color = False
is_thresholding = False
is_gaussian_blurring = False
is_sharpening = False
rotate_angle = 0
video_filename = None
video_out = None
zoom_factor = 1.0
zoom_range = (0.5, 1.0)  # zoom factor range from 1 to 0.5 or 1 to 5 (5x zoom in to no zoom)
guassian_blur_sigma = (5, 5)  # default sigma values for Gaussian blur


btn_zoom_out_pos = None
btn_zoom_in_pos = None


# function check if save folder exists, if not create it and save the frame in it
def create_save_folder():
    path = os.path.join(CURRENT_DIRECTORY, SAVE_FOLDER_PATH)
    if not os.path.exists(path):
        os.makedirs(path)
    print("Save folder created")    

# generate a file path for the photo or video to be saved in the save folder with the name "frame.jpg" or "video.avi"
def generate_file_path(prefix, extension):
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
    filename = generate_file_path("img", "jpg")
    create_save_folder()
    cv.imwrite(filename, frame)
    print(f"Frame captured and saved as '{filename}'")

# start recording video, create a VideoWriter object and save the video in the save 
# folder with the name "video.avi"
def start_recording(filename):
    create_save_folder()
    global video_out
    video_out = cv.VideoWriter(filename, cv.VideoWriter_fourcc(*'MJPG'), 20.0, (640, 480))

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
    if show:
        # cv.namedWindow(windowName)
        cv.createTrackbar("Sigma X", WINDOW_NAME, 0, 100, update_gaussian_x_blur)
        cv.createTrackbar("Sigma Y", WINDOW_NAME, 0, 100, update_gaussian_y_blur)
    else:
        # Remove the trackbars by setting their positions to 0 and max value to 0
        cv.setTrackbarPos("Sigma X", WINDOW_NAME, 0)
        cv.setTrackbarPos("Sigma Y", WINDOW_NAME, 0)

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
    circle_x = int(track_bar_circle_x_range[0] + normalized_zoom * (track_bar_circle_x_range[1] - track_bar_circle_x_range[0]))

    cv.circle(frame, (circle_x, trackbar_pos_y), trackbar_circle_radius, ZOOM_BTN_TXT_COLOR, -1)

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


def main():
    
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(WINDOW_NAME, mouse_callback)

    # initialize the camera
    cap = cv.VideoCapture(0, cv.CAP_V4L2)

    # set the video format to MJPG
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Cannot open camera")
        exit() 

    while True:   
        global is_recording
        global video_filename
        global video_out     
        global is_extracting_color
        global rotate_angle
        global is_thresholding
        global is_gaussian_blurring
        global guassian_blur_sigma
        global is_sharpening

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
        if is_gaussian_blurring:
            apply_gaussian_blur(frame, guassian_blur_sigma)

        # Sharpen the frame if is_sharpening is True using a simple unsharp masking technique 
        # where the blurred image is subtracted from the original image to get
        if is_sharpening:
            sharpen_frame(frame)

        # Create a copy of the show_frame to add date and time without modifying the original frame that will be displayed
        output_frame = frame.copy()
        add_date_time_to_frame(output_frame)

        key = cv.waitKey(1) & 0xFF
        
        if key == ord('v'):
            if not is_recording:
                is_recording = True
                video_filename = generate_file_path("video", "avi")
                start_recording(video_filename)
                
                print("Recording started...")
            else:
                is_recording = False
                stop_recording()                
                print(f"Recording stopped. Video saved as '{video_filename}'")
    
        elif key == ord('c'):
            flash_screen(frame)
            save_photo(output_frame)
            is_capturing = False
        
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
            is_gaussian_blurring = not is_gaussian_blurring
            if is_gaussian_blurring:
                show_gaussian_blur_trackbar(True)
                print("Applying Gaussian blur to frame...")
            else:
                show_gaussian_blur_trackbar(False)
                print("Stopping Gaussian blur...")
        
        # if pressed key is "s" extract sharpened image
        elif key == ord('s'):
            is_sharpening = not is_sharpening
            if is_sharpening:
                print("Extracting sharpened image...")
            else:
                print("Stopping sharpening...")

        elif key == 27:  # ESC key to exit
            print("Exiting...")
            break

        
        # If recording is active, write the current frame to the video file 
        # and add a recording indicator to the frame
        if is_recording:
            if video_out is not None:
                video_out.write(output_frame)

        # Copy the region of interest (ROI) containing the date and time from the output frame to the show frame at the top-right corner of the frame
        copy_datetime_roi(output_frame, frame)

        # Add the OpenCV image to the top-left corner of the frame if it exists
        add_opencv_image_to_frame(frame)

        # Add a red border to the frame to indicate that the camera is active
        frame = add_border_to_frame(frame)

        # Add a zoom trackbar to the frame
        add_zoom_trackbar(frame)

        # Display the resulting frame
        cv.imshow(WINDOW_NAME, frame)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()





