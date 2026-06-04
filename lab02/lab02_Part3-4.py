import numpy as np
from typing import List
import cv2 as cv
from matplotlib import pyplot as plt

# Rotate image by the specified angle using OpenCV's getRotationMatrix2D and warpAffine functions, and return the resulting image
def rotate_image(image, angle):
    h, w = image.shape[:2]  # Get the height and width of the input image
    center = (w / 2, h / 2)  # Calculate the center
    M = cv.getRotationMatrix2D(center, angle, 1.0)  # Get the rotation matrix for the specified angle and scale

    # Calculate the new width and height of the rotated image to ensure that the entire image 
    # fits within the new dimensions, and adjust the rotation matrix to account for the translation 
    # needed to keep the image centered after rotation
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate the image using the rotation matrix and return the resulting image
    rotated = cv.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))  
    return rotated

# Scale the image by the specified factor using linear interpolation and return the resulting image
def scale_image(image, scale_factor):
    # Scale the image by the specified factor using linear interpolation and return the resulting image
    if scale_factor <= 1:
        interpolation = cv.INTER_AREA  # Use nearest neighbor interpolation for non-positive scale factors
    else:
        interpolation = cv.INTER_LINEAR  # Use linear interpolation for positive scale factors
    res = cv.resize(image,None,fx=scale_factor, fy=scale_factor, interpolation = interpolation)  
    return res

# Apply an affine transformation to the input image using predefined source and destination points, 
# and return the resulting image
def affine_transform(image):
    h, w = image.shape[:2]  # Get the height and width of the input image
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])  # Define the source points for the affine transformation
    pts2 = np.float32([[50, 50], [200, 20], [50, 200]])  # Define the destination points for the affine transformation
    M = cv.getAffineTransform(pts1, pts2)  # Get the affine transformation matrix

    # Transform all 4 corners to find new bounding box
    corners = np.float32([[0,0],[w,0],[0,h],[w,h]]).reshape(-1,1,2)
    transformed = cv.transform(corners, M).reshape(-1, 2)

    # Calculate the new width and height of the transformed image to ensure that 
    # the entire image fits within the new dimensions, and adjust the transformation matrix 
    # to account for the translation needed to keep the image centered after transformation
    x_min, y_min = transformed.min(axis=0)
    x_max, y_max = transformed.max(axis=0)

    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    # Shift so no corner goes negative
    M[0, 2] -= x_min
    M[1, 2] -= y_min

    # Apply the affine transformation to the image and return the resulting image
    transformed = cv.warpAffine(image, M, (new_w, new_h), borderValue=(255, 255, 255))  
    return transformed

# Apply a perspective transformation to the input image using predefined source and destination points, 
# and return the resulting image
def perspective_transform(image):
    h, w = image.shape[:2]  # Get the height and width of the input image
    # pst1 is a cross shape
    pts1 = np.float32([[0, 0], [550, 50], [50, 300], [550, 300]])  # Define the source points for the perspective transformation
    pts2 = np.float32([[0, 0], [550, 50], [50, 300], [550, 350]])  # Define the destination points for the perspective transformation
    M = cv.getPerspectiveTransform(pts1, pts2)  # Get the perspective transformation matrix
    
    # Transform all 4 corners
    corners = np.float32([[0,0],[w,0],[0,h],[w,h]]).reshape(-1,1,2)
    transformed = cv.perspectiveTransform(corners, M).reshape(-1, 2)

    # Calculate the new width and height of the perspective transformed image to ensure that the entire image 
    # fits within the new dimensions, and adjust the perspective transformation matrix to account for the translation 
    # needed to keep the image centered after transformation
    x_min, y_min = transformed.min(axis=0)
    x_max, y_max = transformed.max(axis=0)

    new_w = int(x_max - x_min)
    new_h = int(y_max - y_min)

    # Shift translation into M (row 2 is homogeneous)
    T = np.array([[1, 0, -x_min],
                  [0, 1, -y_min],
                  [0, 0,  1   ]], dtype=np.float64)

    M = T @ M   # apply shift after perspective

    # Apply the perspective transformation to the image and return the resulting image
    transformed = cv.warpPerspective(image, M, (new_w, new_h), borderValue=(255, 255, 255))  
    return transformed


# Combine the original and transformed images into a single image for display, and return the resulting combined image
def combine_images(images):

    row1 = np.hstack((images[0], images[1], images[2]))  # Combine the original and transformed images into a single image for display
    row2 = np.hstack((images[3], images[4], images[5]))
    combined_image = np.vstack((row1, row2))  # Combine the two rows
    return combined_image  # Return the combined image with the original and transformed images in a single image for display

# resize the image to the same size as the first image in the list of images, and return the resulting image
def convert_images_to_same_size(images:List):
    new_w, new_h = images[0].shape[:2] # Get the width and height of the first image in the list to use as the new dimensions for all images
    for i in range(1, len(images)):
        h, w = images[i].shape[:2]
        if w > new_w:
            new_w = w
        if h > new_h:
            new_h = h

    for i in range(len(images)):
        images[i] = convert_same_size(new_w, new_h, images[i])  # Convert each image in the list to the same size as the first image using the convert_same_size function
        images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)  # Convert each image in the list from BGR color format to RGB color format for proper display using Matplotlib
    return images  # Return the list of converted images with the same size

# Convert the image to new size
def convert_same_size(new_w, new_h, image):
       
    res = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255  # Create a blank image with the same shape as the original image to store the original frame
    h, w = image.shape[:2]  # Get the height and width of the input image
    # copy scaleddown image to the center of the original frame
    x_offset = (new_w - w) // 2  # Calculate the x offset to center the scaled down image in the original frame
    y_offset = (new_h - h) // 2  # Calculate the y offset to center the scaled down image in the original frame
    res[y_offset:y_offset+h, x_offset:x_offset+w] = image  # Copy the scaled down image to the center of the original frame using the calculated offsets and return the resulting image
    return res

# Add caption to the bottom of the image, and return the resulting image
def add_caption(images, titles):
    font = cv.FONT_HERSHEY_PLAIN  # Define the font to use for the title text
    font_scale = 2  # Set the font scale for the title text
    thickness = 2  # Set the thickness of the title text
    color = (0, 0, 0)  # Set the color of the title text to black
    padding = 10  # Set the padding between the title text and the bottom of the image
    for i in range(len(images)):
        text_size, _ = cv.getTextSize(titles[i], font, font_scale, thickness)  # Get the size of the title text using the specified font and scale
        text_x = (images[i].shape[1] - text_size[0]) // 2  # Calculate the x coordinate to center the title text horizontally in the image
        text_y = images[i].shape[0] - text_size[1] - padding  # Calculate the y coordinate to center the title text vertically in the image
        cv.putText(images[i], titles[i], (text_x, text_y), font, font_scale, color, thickness)  # Add the title text to the image at the calculated coordinates with the specified font, scale, color, and thickness
    return images  # Return the list of images with the added title text

# Detect corners using Harris corner detection, 
# apply the Harris corner detection algorithm to the input image, 
# and return the resulting image with detected corners highlighted
def harris_corner_detection(image):
    # Convert the input image to grayscale for corner detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  
    # Convert the grayscale image to float32 format for processing by the Harris corner detection algorithm
    gray = np.float32(gray)  
     # Apply the Harris corner detection algorithm to the grayscale image
    dst = cv.cornerHarris(gray, 2, 3, 0.04) 
    # Dilate the corner response image to enhance the corner points for better visualization
    dst_dilated = cv.dilate(dst, None)  

    # Highlight the detected corners in the original image by setting the pixel values to red f
    # or crosses where the corner response is above a certain threshold (in this case, 4% of the maximum response value)
    image[dst_dilated > 0.04 * dst_dilated.max()] = [0, 0, 255]  

    return image

def harris_corner_detection_batch(images):
    res = [None] * len(images)
    for i in range(len(images)):
        res[i] = harris_corner_detection(images[i].copy())
    return res

# Detect SIFT features in the input image, draw the detected keypoints on the original image for visualization, 
# and return the resulting image with detected keypoints highlighted
def sift_feature_detection(image):
     # Convert the input image to grayscale for feature detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) 
    # Create a SIFT feature detector object
    sift = cv.SIFT_create(
        nfeatures=200,
        contrastThreshold=0.05, 
    )  
    # Detect SIFT features in the grayscale image and store the resulting keypoints in the variable keypoints
    keypoints = sift.detect(gray, None)
    # Set the color for drawing the detected keypoints to green
    color = (0, 255, 0)  
    # Draw the detected keypoints on the original image for visualization
    transformed = cv.drawKeypoints(image, keypoints, None, color) 
    return transformed

def sift_feature_detection_batch(images):
    res = [None] * len(images)
    for i in range(len(images)):
        res[i] = sift_feature_detection(images[i].copy())
    return res

def main():
    image = cv.imread('./UnityHall.png')  # Read the image file 'lena.png' and store it in the variable img
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Rotate the image by 10 degrees
    rotated_img = rotate_image(image, 10) 

    # Scale up 20% of the original image
    scaledUp_img = scale_image(image, 1.2) 

    # Scale down 20% of the original image
    scaledDown_img = scale_image(image, 0.8)

    # Apply an affine transformation to the original image
    affine_img = affine_transform(image)  

    # Apply a perspective transformation to the original image
    perspective_img = perspective_transform(image)

    images = convert_images_to_same_size([image, scaledUp_img, affine_img, rotated_img, scaledDown_img, perspective_img])  
    captions = ['Original', 'Scaled Up', 'Affine', 'Rotated', 'Scaled Down', 'Perspective'] 
    
    # Harris corner detection for all images
    # Add caption to the bottom of the image, and return the resulting image
    harris_detected_corners = harris_corner_detection_batch(images) 
    harris_detected_corners = add_caption(harris_detected_corners, captions)
    combined_harris = combine_images(harris_detected_corners)
    cv.imshow('Harris Corner Detected Image', combined_harris)

    # SIFT feature detection for all images
    sift_detected_features = sift_feature_detection_batch(images)
    sift_detected_features = add_caption(sift_detected_features, captions)
    combined_sift = combine_images(sift_detected_features)
    cv.imshow('SIFT Feature Detected Image', combined_sift)

    cv.waitKey(0)  # Wait for a key press to close the displayed image windows
    cv.destroyAllWindows()  # Destroy all OpenCV windows to free up resources
    
if __name__ == "__main__":
    main()