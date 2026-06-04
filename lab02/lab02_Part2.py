import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

WINDOW_NAME = "Quan Vision-Lab2 Part2"

# Convolution function to apply a 3x3 kernel to a grayscale image, used for edge detection filters
def _convolve3x3(gray_frame, kernel):
    h, w = gray_frame.shape
    out = np.zeros((h, w), dtype=np.float32)
    for ki in range(3):
        for kj in range(3):
            if kernel[ki, kj] != 0:
                out[1:h-1, 1:w-1] += kernel[ki, kj] * gray_frame[ki:h-2+ki, kj:w-2+kj]
    return out

# Sobel filter without using OpenCV's built-in functions, apply the Sobel filter to the 
# input frame using the specified kernel size and update the frame with the resulting image
def sobel_filter(frame, direction=None):
    direction = direction.lower() if direction else None

    # Define the Sobel kernels for x and y directions, used for edge detection in the respective directions
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # Convert the input frame to grayscale and apply the Sobel filter in the specified direction, 
    # then compute the magnitude of the gradient and return the resulting image
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
    Gx = _convolve3x3(gray_frame, sobel_x) if direction == 'x' else np.zeros_like(gray_frame)
    Gy = _convolve3x3(gray_frame, sobel_y) if direction == 'y' else np.zeros_like(gray_frame)

    # Compute the magnitude of the gradient using the Sobel filter results in both x and y directions,
    G = np.sqrt(Gx**2 + Gy**2)

    # Clip the resulting gradient magnitude to the range [0, 255] and convert it to uint8 format for display
    G_clipped = np.clip(G, 0, 255).astype(np.uint8)
    
    return cv.cvtColor(G_clipped, cv.COLOR_GRAY2BGR)

# Laplacian filter without using OpenCV's built-in functions, apply the Laplacian filter 
# to the input frame using a predefined kernel and return the resulting image
def laplacian_filter(frame):
    # Define the Laplacian kernel, used for edge detection by calculating the second derivative of the image intensity
    laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    # Convert the input frame to grayscale and apply the Laplacian filter using the defined kernel, 
    # then clip the resulting values to the range [0, 255] and convert it to uint8 format for display
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = np.clip(_convolve3x3(gray_frame, laplacian_kernel), 0, 255).astype(np.uint8)

    return cv.cvtColor(laplacian, cv.COLOR_GRAY2BGR)

# Function to apply the Sobel and Laplacian filters to the input frame, display the original and 
# filtered images in a 2x2 grid using Matplotlib, and pause briefly to allow the plots to update
def plot_filters(frame):
    # Apply the Sobel filter in both x and y directions to the input frame to detect edges.
    frame_SobelX = sobel_filter(frame, direction='x')  # Apply the Sobel filter to the frame to detect edges
    frame_SobelY = sobel_filter(frame, direction='y')  # Apply the Sobel filter to the frame to detect edges

    # Apply the Laplacian filter to the input frame to detect edges and store the resulting image in frame_laplacian
    frame_laplacian = laplacian_filter(frame)  # Apply the Laplacian filter to the frame to detect edges
    
    # Create a figure with a specified size to display the original and filtered images, 
    # and arrange them in a 2x2 grid using subplots.
    plt.figure(figsize=(8, 6))  # Create a figure with a specified size to display the original and filtered images
    plt.subplot(2, 2, 1)
    plt.imshow(frame)  # Display the original frame in the first subplot)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(frame_laplacian)  # Display the Laplacian filtered frame in the second subplot
    plt.title('Laplacian')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(frame_SobelX)  # Display the Sobel X filtered frame in the third subplot
    plt.title('Sobel X')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(frame_SobelY)  # Display the Sobel Y filtered frame in the fourth subplot
    plt.title('Sobel Y')
    plt.axis('off')

    print("Plotting filters...")  # Print a message to indicate that the filters are being displayed
    plt.tight_layout()  # Adjust the layout of the subplots to prevent overlap
    plt.show()  # Show the plots without blocking the main threads

def main():
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_AUTOSIZE)
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_BUFFERSIZE, 10)

    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        cv.imshow(WINDOW_NAME, frame)

        # Wait for a key press and check if the '4' key is pressed to apply filters and display the results,
        key = cv.waitKey(1) & 0xFF
        if key == ord('4'):
            # Call the function to apply filters and display the results when '4' key is pressed
            plot_filters(frame)

        if key == 27:  # ESC key to exit
            break

if __name__ == "__main__":
    main()