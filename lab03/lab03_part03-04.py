import cv2 as cv
import os
import numpy as np


SIFT_CONTROLS_WINDOWNAME = "SIFT Controls"
SURF_CONTROLS_WINDOWNAME = "SURF Controls"
CURRENT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

nfeatures = 500
contrast_threshold = 0.04
hessian_threshold = 400
nOctaves = 4
nOctaveLayers = 3
extended = False

current_settings = (nfeatures, contrast_threshold, hessian_threshold, nOctaves, nOctaveLayers, extended)
previous_settings = None

# Detect keypoints and compute descriptors using the specified detector (SIFT or SURF)
# and return them as a tuple
def detectAndCompute(detector, img):
    kp, des = detector.detectAndCompute(img,None)
    return kp, des

# Brute-force matcher for SIFT/SURF, using L2 norm and no cross-checking, and return the matches
def brute_force_matcher(desc1, desc2, k=2):
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    return bf.knnMatch(desc1, desc2, k)

# FLANN matcher for SIFT/SURF, using KD-tree algorithm, and return the matches
def FLANN_matcher(desc1, desc2, k=2):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    return flann.knnMatch(desc1, desc2, k)

# Deaw matches between two images using the provided keypoints and matches, and return the resulting image
def draw_matches(img1, kp1, img2, kp2, matches, ratio=0.7, matchColor=(0, 255, 0)):
    matchesMask = [[0,0] for i in range(len(matches))]
    for i,(m,n) in enumerate(matches):
        if m.distance < ratio*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = matchColor,
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

# Resize the image by the specified scale factor and return the resulting image
def resize_by_scale(img, scale=0.5):
    return cv.resize(img, (0,0), fx=scale, fy=scale)

# Update the parameters of the SIFT detector with the specified values
def update_SIFT_parameters(sift, nfeatures, contrast_threshold):
    print_SIFT_parameters(sift)  # Print the current SIFT parameters before updating
    sift.setNFeatures(nfeatures)
    sift.setContrastThreshold(contrast_threshold)

# Update the parameters of the SURF detector with the specified values
def update_SURF_parameters(surf, hessian_threshold, nOctaves, nOctaveLayers, extended):
    print_SURF_parameters(surf)  # Print the current SURF parameters before updating
    surf.setHessianThreshold(hessian_threshold)
    surf.setNOctaves(nOctaves)
    surf.setNOctaveLayers(nOctaveLayers)
    surf.setExtended(extended)

# Print the current parameters of the SIFT and SURF detectors for debugging purposes
def print_SIFT_parameters(sift):
    print(f"SIFT Parameters: nFeatures={sift.getNFeatures()}, contrastThreshold={sift.getContrastThreshold()}")
def print_SURF_parameters(surf):
    print(f"SURF Parameters: hessianThreshold={surf.getHessianThreshold()}, nOctaves={surf.getNOctaves()}, nOctaveLayers={surf.getNOctaveLayers()}, extended={surf.getExtended()}")

# Save the current settings of the SIFT and SURF parameters in a global variable for later comparison
def save_settings():
    global current_settings
    current_settings = (nfeatures, contrast_threshold, hessian_threshold, nOctaves, nOctaveLayers, extended)

# Callback functions for trackbar changes to update the corresponding parameters and save the current settings
def onchange_nFeatures(x):
    global nfeatures
    nfeatures = x
    save_settings()
def onchange_contrast_threshold(x):
    global contrast_threshold
    contrast_threshold = x / 10
    save_settings()
def onchange_hessian_threshold(x):
    global hessian_threshold
    hessian_threshold = x
    save_settings()
def onchange_nOctaves(x):
    global nOctaves
    nOctaves = x
    save_settings()
def onchange_nOctaveLayers(x):
    global nOctaveLayers
    nOctaveLayers = x
    save_settings()
def onchange_extended(x):
    global extended
    extended = bool(x)
    save_settings()

# Check if the SIFT or SURF parameters have changed by comparing the current settings with the previous settings,
# and return True if any parameter has changed, otherwise return False
def is_sift_parameters_changed(sift):
    return (sift.getNFeatures() != nfeatures) or (sift.getContrastThreshold() != contrast_threshold)
def is_surf_parameters_changed(surf):
    return (surf.getHessianThreshold() != hessian_threshold) or (surf.getNOctaves() != nOctaves) or (surf.getNOctaveLayers() != nOctaveLayers) or (surf.getExtended() != extended)

def add_caption(image, title):
    font = cv.FONT_HERSHEY_PLAIN
    font_scale = 2
    thickness = 2
    color = (0, 0, 255)
    text_x = 20
    text_y = 450
    image = cv.putText(image, title, (text_x, text_y), font, font_scale, color, thickness)
    # return image

# Create a control window with trackbars for SIFT and SURF parameters
def create_control_window():

    cv.imshow(SIFT_CONTROLS_WINDOWNAME, np.zeros((1, 400), dtype=np.uint8))
    cv.createTrackbar("nFeatures", SIFT_CONTROLS_WINDOWNAME, nfeatures, 1000, onchange_nFeatures)
    cv.createTrackbar("Contrast Threshold", SIFT_CONTROLS_WINDOWNAME, int(contrast_threshold * 500), 500, onchange_contrast_threshold)

    cv.imshow(SURF_CONTROLS_WINDOWNAME, np.zeros((1, 400), dtype=np.uint8))
    cv.createTrackbar("Hessian Threshold", SURF_CONTROLS_WINDOWNAME, hessian_threshold, 10000, onchange_hessian_threshold)
    cv.createTrackbar("nOctaves", SURF_CONTROLS_WINDOWNAME, nOctaves, 10, onchange_nOctaves)
    cv.createTrackbar("nOctaveLayers", SURF_CONTROLS_WINDOWNAME, nOctaveLayers, 10, onchange_nOctaveLayers)
    cv.createTrackbar("Extended", SURF_CONTROLS_WINDOWNAME, int(extended), 1, onchange_extended)

def main():
    global nfeatures, contrast_threshold, hessian_threshold, nOctaves, nOctaveLayers, extended, current_settings, previous_settings
    
    # Load images and resize for faster processing
    root = os.getcwd()
    img_book = cv.imread(os.path.join(CURRENT_DIRECTORY, 'book.png'), cv.IMREAD_GRAYSCALE)
    img_book = resize_by_scale(img_book, 0.6)  # resize for better display
    img_scene = cv.imread(os.path.join(CURRENT_DIRECTORY, 'book_in_scene.png'), cv.IMREAD_GRAYSCALE)
    img_scene = resize_by_scale(img_scene, 0.75)  # resize for better display

    # Create sift and surf detectors
    sift = cv.SIFT_create()
    surf = cv.xfeatures2d.SURF_create(400)

    # Pre-computed kp/des — updated when settings change, then passed to compare_detectors
    sift_kp1 = sift_des1 = sift_kp2 = sift_des2 = None
    surf_kp1 = surf_des1 = surf_kp2 = surf_des2 = None

    initial = True # Flag to indicate the first iteration for creating control windows and trackbars
    display_scale = 0.5 # Scale factor for displaying match images (adjust as needed)

    create_control_window()  # Create control windows and trackbars for SIFT and SURF parameters
    
    while True:
        if current_settings != previous_settings:

            if initial:
                print("Initial settings:")
                print(f"  SIFT - nfeatures: {nfeatures}, contrast_threshold: {contrast_threshold / 100}")
                print(f"  SURF - hessian_threshold: {hessian_threshold}, nOctaves: {nOctaves}, nOctaveLayers: {nOctaveLayers}, extended: {extended}")

            if is_sift_parameters_changed(sift) or initial:
                # print("SIFT parameters changed, updating matches...")
                update_SIFT_parameters(sift, nfeatures, contrast_threshold / 100)
                # SIFT    
                sift_kp1, sift_des1 = detectAndCompute(sift, img_book)
                sift_kp2, sift_des2 = detectAndCompute(sift, img_scene)

                # SIFT and Brute-Force
                sift_brute_matches = brute_force_matcher(sift_des1, sift_des2)
                sift_brute_img = draw_matches(img_book, sift_kp1, img_scene, sift_kp2, sift_brute_matches, matchColor=(255, 255, 0))
                sift_brute_img = resize_by_scale(sift_brute_img, display_scale)  # resize for better display
                add_caption(sift_brute_img, "SIFT + Brute-Force")
                cv.imshow('SIFT Matches (Brute-Force)', sift_brute_img)  # Display SIFT + Brute-Force matches in a window

                # SIFT and FLANN
                sift_flann_matches = FLANN_matcher(sift_des1, sift_des2)
                sift_flann_img = draw_matches(img_book, sift_kp1, img_scene, sift_kp2, sift_flann_matches)
                sift_flann_img = resize_by_scale(sift_flann_img, display_scale)  # resize for better display
                add_caption(sift_flann_img, "SIFT + FLANN")
                cv.imshow('SIFT Matches (FLANN)', sift_flann_img)  # Display SIFT + FLANN matches in a window
                
                # sift_combined_img = np.hstack((sift_brute_img, sift_flann_img))  # Combine SIFT match images for display
                # cv.imshow('SIFT Matches (Brute-Force | FLANN)', sift_combined_img)
            
            if is_surf_parameters_changed(surf) or initial:                
                # print("SURF parameters changed, updating matches...")
                update_SURF_parameters(surf, hessian_threshold, nOctaves, nOctaveLayers, extended)
                # SURF
                surf_kp1, surf_des1 = detectAndCompute(surf, img_book)
                surf_kp2, surf_des2 = detectAndCompute(surf, img_scene)

                # SURF and Brute-Force
                surf_brute_matches = brute_force_matcher(surf_des1, surf_des2)
                surf_brute_img = draw_matches(img_book, surf_kp1, img_scene, surf_kp2, surf_brute_matches, matchColor=(255, 255, 0))
                surf_brute_img = resize_by_scale(surf_brute_img, display_scale)  # resize for better display
                add_caption(surf_brute_img, "SURF + Brute-Force")
                cv.imshow('SURF Matches (Brute-Force)', surf_brute_img)  # Display SURF + Brute-Force matches in a window

                # SURF and FLANN
                surf_flann_matches = FLANN_matcher(surf_des1, surf_des2)
                surf_flann_img = draw_matches(img_book, surf_kp1, img_scene, surf_kp2, surf_flann_matches)
                surf_flann_img = resize_by_scale(surf_flann_img, display_scale)  # resize for better display
                add_caption(surf_flann_img, "SURF + FLANN")
                cv.imshow('SURF Matches (FLANN)', surf_flann_img)  # Display SURF + FLANN matches in a window

                # surf_combined_img = np.hstack((surf_brute_img, surf_flann_img))  # Combine SURF match images for display
                # cv.imshow('SURF Matches (Brute-Force | FLANN)', surf_combined_img)

            previous_settings = current_settings
            initial = False
        
        key = cv.waitKey(100)
        if key == 27:  # ESC key to exit
            cv.destroyAllWindows()
            break

if __name__ == "__main__":
    main()