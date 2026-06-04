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


# Show 2x3 grid of original and transformed images, and print a message to indicate that the images are being displayed
def show_transformed_images(images, captions):
       
    # add caption to each image
    for i in range(len(images)):
        images[i] = add_caption(images[i], captions[i])

    # convert to RGB
    for i in range(len(images)):
        images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)

    row1 = np.hstack((images[0], images[1], images[2]))  # Combine the original and transformed images into a single image for display
    row2 = np.hstack((images[3], images[4], images[5]))
    combined_image = np.vstack((row1, row2))  # Combine the two rows

    print("Showing transformed images...")  # Print a message to indicate that the images are being displayed
    cv.imshow('Transformed Images', combined_image)  # Display the combined image with the original and transformed images in a window titled 'Transformed Images'

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
def add_caption(image, title):
    font = cv.FONT_HERSHEY_PLAIN  # Define the font to use for the title text
    font_scale = 2  # Set the font scale for the title text
    thickness = 2  # Set the thickness of the title text
    color = (0, 0, 0)  # Set the color of the title text to black
    padding = 10  # Set the padding between the title text and the bottom of the image
    text_size, _ = cv.getTextSize(title, font, font_scale, thickness)  # Get the size of the title text using the specified font and scale
    text_x = (image.shape[1] - text_size[0]) // 2  # Calculate the x coordinate to center the title text horizontally in the image
    text_y = image.shape[0] - text_size[1] - padding  # Calculate the y coordinate to center the title text vertically in the image
    cv.putText(image, title, (text_x, text_y), font, font_scale, color, thickness)  # Add the title text to the image at the calculated coordinates with the specified font, scale, color, and thickness
    return image  # Return the image with the added title text

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

    images = convert_images_to_same_size([image, scaledUp_img, affine_img, rotated_img, scaledDown_img, perspective_img])  # Convert all images to the same size for consistent display
    captions = ['Original', 'Scaled Up', 'Affine', 'Rotated', 'Scaled Down', 'Perspective']  # Define the titles for each image to be displayed
 
    show_transformed_images(images, captions)  # Call the function to show the original and transformed images in a single window
    
    cv.waitKey(0)  # Wait for a key press to close the displayed image windows
    cv.destroyAllWindows()  # Destroy all OpenCV windows to free up resources
    
if __name__ == "__main__":
    main()