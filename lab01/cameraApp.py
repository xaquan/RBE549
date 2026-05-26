from fileinput import filename
import cv2 as cv
import os
from datetime import datetime

CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

# Define constants for the save folder path, zoom button properties, trackbar padding, and datetime position
SAVE_FOLDER_PATH = "saved_media"    
ZOOM_BTN_WIDTH = 20
ZOOM_BTN_COLOR = (200, 200, 200)  # gray color for buttons
ZOOM_BTN_TXT_COLOR = (0, 0, 0)  # black color for text
TRACKBAR_PADDING_LR = 80 
TRACKBAR_PADDING_BOTTOM = 50

DATETIME_POS = (470, 470)

OPENCV_IMG_PATH = os.path.join(CURRENT_DIRECTORY, "openCV.png")


is_recording = False
video_filename = None
video_out = None
zoom_factor = 1.0
zoom_range = (0.5, 1.0)  # zoom factor range from 1 to 0.5 or 1 to 5 (5x zoom in to no zoom)


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
    flash_frame = frame.copy()
    flash_frame[:] = (255, 255, 255)
    cv.imshow("xCamera", flash_frame)
    cv.waitKey(100)  # wait for 100 milliseconds
    

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
        return frame

    height, width = frame.shape[:2]
    center_x, center_y = width // 2, height // 2

    new_width = int(width * zoom_factor)
    new_height = int(height * zoom_factor)

    x1 = max(0, center_x - new_width // 2)
    y1 = max(0, center_y - new_height // 2)
    x2 = min(width, center_x + new_width // 2)
    y2 = min(height, center_y + new_height // 2)

    zoomed_frame = frame[y1:y2, x1:x2]
    return cv.resize(zoomed_frame, (width, height), interpolation=cv.INTER_LINEAR)

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
        frame[0:img_height, 0:img_width] = opencv_img

def main():
    window_name = "xCamera"
    
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback(window_name, mouse_callback)

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

        # Capture frame-by-frame
        ret, show_frame = cap.read()        

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Apply zoom to the frame based on the current zoom factor and display it
        show_frame = apply_zoom(show_frame, zoom_factor)

        # Create a copy of the show_frame to add date and time without modifying the original frame that will be displayed
        output_frame = show_frame.copy()            
        add_date_time_to_frame(output_frame)
        
        # Wait for a key press for 1 millisecond and check if the 'v' key is pressed to start or stop recording video, if the 'c' key is pressed to capture a photo, or if the 'esc' key is pressed to exit the loop
        # key = cv.waitKey(1) & 0xFF

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
            flash_screen(show_frame)
            save_photo(output_frame)
            is_capturing = False
        
        # if pressed key is "e", extract pink
        elif key == ord('e'):
            print("Extracting pink color...")
        
        # if pressed key is "r", rotate by 10 degrees
        elif key == ord('r'):
            print("Rotating frame by 10 degrees...")
        
        # if pressed key is "t", translate the frame by 10 pixels to the right and 10 pixels down
        elif key == ord('t'):
            print("Translating frame by 10 pixels to the right and 10 pixels down...")
        
        # if pressed key is "b", show trackbar to control gausian blur
        elif key == ord('b'):
            print("Showing trackbar to control Gaussian blur...")
        
        # if pressed key is "s" extract sharpened image
        elif key == ord('s'):
            print("Extracting sharpened image...")

        elif key == 27:  # ESC key to exit
            print("Exiting...")
            break
        
        # If recording is active, write the current frame to the video file 
        # and add a recording indicator to the frame
        if is_recording:
            if video_out is not None:
                video_out.write(output_frame)

        # Copy the region of interest (ROI) containing the date and time from the output frame to the show frame at the top-right corner of the frame
        copy_datetime_roi(output_frame, show_frame)

        # Add the OpenCV image to the top-left corner of the frame if it exists
        add_opencv_image_to_frame(show_frame)

        # Add a red border to the frame to indicate that the camera is active
        show_frame = add_border_to_frame(show_frame)

        # Add a zoom trackbar to the frame
        add_zoom_trackbar(show_frame)

        # Display the resulting frame
        cv.imshow(window_name, show_frame)
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()





