import cv2 as cv
import os
import numpy as np

root = os.getcwd()
reference = cv.imread(os.path.join(root, 'Assign2/reference/ref01.png'), cv.IMREAD_GRAYSCALE)

sift = cv.SIFT_create()
kp2, des2 = sift.detectAndCompute(reference,None)

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*"MJPG"))
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)

sift_threshold = 0.04  # Default threshold for SIFT matching
window_name = 'SIFT Coin Detection'
cv.namedWindow(window_name)
tb_sift_threshold = cv.createTrackbar('SIFT Threshold', window_name, int(sift_threshold * 500), 500, lambda x: None)

reference_circles = cv.HoughCircles(reference, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)

# keypoints and descriptors for the reference circles
reference_kps = []
reference_deses = []
for circle in reference_circles[0, :]:
    x, y, r = circle
    mask = np.zeros(reference.shape[:2], dtype=np.uint8)
    cv.circle(mask, (int(x), int(y)), int(r), 255, thickness=-1)
    kp, des = sift.detectAndCompute(reference, mask)
    reference_kps.append(kp)
    reference_deses.append(des)

while True:
    ret, coin = cap.read()
    if not ret:
        break

    coin_gray = cv.cvtColor(coin, cv.COLOR_BGR2GRAY)
    sift_threshold = cv.getTrackbarPos('SIFT Threshold', window_name) / 500.0
    sift.setContrastThreshold(sift_threshold)

    kp1, des1 = sift.detectAndCompute(coin,None)
    if des1 is None or des2 is None:
        continue

    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    # Match descriptors between the current frame and the reference kps and deses
    matches = []
    for des2 in reference_deses:
        if des2 is not None:
            matches.append(bf.match(des1, des2))

    # draw best matches with a reference circle
    if matches:
        best_matches = min(matches, key=lambda x: len(x))
        best_index = matches.index(best_matches)
        kp2 = reference_kps[best_index]
        des2 = reference_deses[best_index]

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = cv.drawMatches(coin, kp1, reference, kp2, matches[:10], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv.imshow(window_name, img3)


    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cv.destroyAllWindows()
