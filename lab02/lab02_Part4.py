import cv2 as cv
from matplotlib.pyplot import gray
import numpy as np

WINDOW_NAME = "Quan Vision-Lab2 Part4"

k = 0.04  # Set the initial value of the k parameter for the Harris corner detection algorithm

# Detect corners using Harris corner detection, 
# apply the Harris corner detection algorithm to the input image, 
# and return the resulting image with detected corners highlighted
def harris_corner_detection(image, k=0.04):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert the input image to grayscale for corner detection
    # gray = np.float32(gray)  # Convert the grayscale image to float32 format for processing by the Harris corner detection algorithm

    dst = cv.cornerHarris(gray, 2, 3, k)  # Apply the Harris corner detection algorithm to the grayscale image
    # cv.imshow('Harris Corners algorithm', dst)  # Display the resulting image with detected corners in a window titled 'Harris Corners'
    
    dst_dilated = cv.dilate(dst, None)  # Dilate the corner response image to enhance the corner points for better visualization
    # cv.imshow('Dilated Corners', dst_dilated)  # Display the dilated corner response image in a window titled 'Dilated Corners'
    
    # Highlight the detected corners in the original image by setting the pixel values to red for crosses where the corner response is above a certain threshold
    image[dst_dilated > 0.01 * dst_dilated.max()] = [0, 0, 255]  # Set the pixel values to red for corners where the dilated corner response is above 1% of the maximum response value
    # cv.imshow(WINDOW_NAME, image)  # Display the image with detected corners in a window titled 'Harris Corners'
    
   
    # corner_coords = np.argwhere(dst_dilated > 0.01 * dst_dilated.max())
    # for y, x in corner_coords:
    #     cv.drawMarker(image, (x, y), color=(0, 0, 255),
    #                 markerType=cv.MARKER_CROSS,
    #                 markerSize=4,
    #                 thickness=1)
    return image

def sift_feature_detection(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Convert the input image to grayscale for feature detection
    sift = cv.SIFT_create()  # Create a SIFT feature detector object
    keypoints = sift.detect(gray, None)  # Detect SIFT features in the grayscale image and store the resulting keypoints in the variable keypoints
    # image = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)  # Convert the grayscale image back to BGR color format for visualization of detected features
    color = (0, 255, 0)  # Set the color for drawing the detected keypoints to green
    transformed = cv.drawKeypoints(image, keypoints, None, color)  # Draw the detected keypoints on the original image for visualization
    return transformed  # Return the image with detected keypoints highlighted

def update_k(x):
    global k  # Declare k as a global variable to allow it to be modified within the function
    k = x / 1000.0  # Update the value of k based on the position of the trackbar, scaling it down by a factor of 100 for use in the Harris corner detection algorithm
    print(f"Updated k: {k}")  # Print the updated value of k to

def main():
    image = cv.imread('UnityHall.png')  # Read the input image from the specified file path
    cv.namedWindow(WINDOW_NAME)  # Create a window with the specified name for displaying the results of the Harris corner detection algorithm
    if image is None:
        print("Error: Could not read image")
        return
    
    
    # image = cv.cvtColor(image, cv.COLOR_BGR2)  # Convert the color space of the image from BGR to RGB for proper display using Matplotlib
    cv.createTrackbar('k', WINDOW_NAME, int(k * 1000), 300, update_k)  # Create a trackbar named 'k' in the 'Harris Corners' window to allow the user to adjust the value of the k parameter for the Harris corner detection algorithm
    # convert k to 2 decimal places
    
    while True:
    
        # global k  # Declare k as a global variable to allow it to be modified within the loop
        harris_corners = harris_corner_detection(image.copy(), k)  # Call the harris_corner_detection function to detect corners in the input image and store the resulting image in the variable harris_corners
        sift_features = sift_feature_detection(image.copy())  # Call the sift_feature_detection function to detect SIFT features in the input image and store the resulting image in the variable sift_features

        combined_image = np.hstack((harris_corners.copy(), sift_features.copy()))  # Combine the images with detected corners and SIFT features side by side for display

        cv.imshow(WINDOW_NAME, combined_image)  # Display the combined image in the window titled 'Harris Corners'
       
        key = cv.waitKey(1)  & 0xFF 

        if key == 27: 
            break  # Exit the loop and end the program

        

    cv.destroyAllWindows()  # Close all OpenCV windows after the loop ends

if __name__ == "__main__":
    main()