from dataclasses import dataclass

import cv2 as cv
from cv2.gapi import mask
from matplotlib.pyplot import hsv
import numpy as np
import os

@dataclass
class CoinValueByRadius:
    radius_min: int
    radius_max: int
    value: float
    name: str = ""  # Optional name for the coin type

CURRENT_DIR = os.getcwd()
DEFAULT_EXPOSURE = 180  # Default exposure value for the camera
PENNY_COLOR_LOWER = np.array([5, 70, 100])  # Lower bound for penny color in HSV
PENNY_COLOR_UPPER = np.array([17, 150, 255]) # Upper bound for penny color in HSV
COINTYPE_DATABASE = [
    CoinValueByRadius(radius_min=52, radius_max=58,value=0.25, name="Quarter"),
    CoinValueByRadius(radius_min=38, radius_max=42, value=0.10, name="Dime"),
    CoinValueByRadius(radius_min=46, radius_max=51, value=0.05, name="Nickel"),
    CoinValueByRadius(radius_min=43, radius_max=45, value=0.01, name="Penny"),
]

circles_prev = []

# Detect circles in the frame using HoughCircles
def detect_circle(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    rows = gray.shape[0]
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, rows / 8,
                               param1=150, param2=50,
                               minRadius=35, maxRadius=65)
    return circles

# Draw the detected circles on the frame
def draw_circles_outline(frame, circles, color=(0, 255, 0), thickness=2):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            cv.circle(frame, center, radius, color, thickness)
            # cv.circle(frame, center, 2, (0, 0, 255), 3)

# Print circles information in one line, sort by location
def print_circles_info(frame, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        sorted_circles = sorted(circles[0, :], key=lambda x: (x[1], x[0]))  # Sort by y, then x
        print("Detected circles (sorted by location):")
        for circle in sorted_circles:            
            color = get_color_of_the_circle(frame, circle)  # Get color of the first circle
            print(f"Center: ({circle[0]}, {circle[1]}), Radius: {circle[2]}, Color: {color}")
    else:
        print("No circles detected.")

# check if the coin is a penny by color
def is_penny_by_color(frame, circle):
    # Convert the image to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)

    circle_color = get_color_of_the_circle(frame, circle)

    # Check if the average color falls within the penny color range
    if np.all(circle_color >= PENNY_COLOR_LOWER) and np.all(circle_color <= PENNY_COLOR_UPPER):
        return True
    return False

# Detect coin value by radius with tolerance
def detect_coin_value_by_radius(frame, circle):
    radius = circle[2]
    result = None
    for coin in COINTYPE_DATABASE:        
        if coin.radius_min <= radius <= coin.radius_max:
            result = coin
            if coin.name == "Dime":       
                # print(f"Dime detected with radius: {radius}, Color: {get_color_of_the_circle(frame, circle)}")           
                if is_penny_by_color(frame, circle):
                    result = COINTYPE_DATABASE[3]  # Penny
            
            return result.value
    print(f"Unknown coin detected with radius: {radius}")
    return 0.0  # Return 0 if no match found

# Calculate the total value of detected coins
def calculate_total_value(frame, circles):
    total_value = 0.0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            radius = circle[2]
            coin_value = detect_coin_value_by_radius(frame, circle)
            total_value += coin_value
    return total_value

# Add total value to the screen
def add_total_to_screen(frame, total_value):
    """
    Add the total value to the screen.
    :param frame: The input image frame (BGR).
    :param total_value: The total value of the coins.
    """
    text = f"Total Value: ${total_value:.2f}"
    cv.putText(frame, text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Get the average color of the circle in HSV
def get_color_of_the_circle(frame, circle):
    """
    Get the average color of the circle in HSV color space.
    :param frame: The input image frame (BGR).
    :param circle: The detected circle (x, y, radius).
    :return: The average HSV color of the circle.
    """
    # Convert the image to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Create a mask for the current circle
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.circle(mask, (circle[0], circle[1]), circle[2], 255, thickness=-1)

    # Get mean HSV values within the circle
    mean_hsv = cv.mean(hsv, mask=mask)[:3]
    # print(f"Mean HSV for circle at ({circle[0]}, {circle[1]}), radius {circle[2]}: {mean_hsv}")

    return mean_hsv

# If the circles total is changed or the poistion of the circles is changed by 5px return True
def is_circles_changed(circles):
    global circles_prev
    if circles is None and circles_prev is None:
        return False
    if (circles is None) != (circles_prev is None):
        circles_prev = circles
        return True
    if len(circles[0]) != len(circles_prev[0]):
        circles_prev = circles
        return True
    for c1, c2 in zip(circles[0], circles_prev[0]):
        if np.linalg.norm(c1[:2] - c2[:2]) > 5:  # Check position change greater than 5 pixels
            circles_prev = circles
            return True
    return False


if __name__ == "__main__":

    print("Starting coin detection...")
    
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    
    window_name = 'VisionCoin'
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)

    calculate_total = None
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        circles = detect_circle(frame)
        # print_circles_info(frame, circles)
        draw_circles_outline(frame, circles)

        if calculate_total is None or is_circles_changed(circles):
            circles_prev = circles  # Update the previous circles for the next iteration
            calculate_total = calculate_total_value(frame, circles)

        add_total_to_screen(frame, calculate_total)

        cv.imshow(window_name, frame)
        # add nFeature and Constart Threshold trackbars to the window

        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC key to exit
            break
    
