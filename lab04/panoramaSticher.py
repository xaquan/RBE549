from unittest import result

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


# Step 0: Declare constants and helper functions
# Step 1: Load images
# Step 2: Detect keypoints and compute descriptors
# Step 3: Match descriptors
# Step 4: Find homography and warp images

MIN_MATCH_COUNT = 10

class PanoramaStitcher:
    def __init__(self):
        self.images = []
        self.stitchedImage = None       
    
    # Function to add images to the stitcher
    def addImage(self, img):
        self.images.append(img)

    def addImages(self, img_list: list):
        self.images.extend(img_list)

    # Helper functions for finding good matches and homography points
    @staticmethod
    def findGoodMatches(matches):
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        return good_matches
    # Helper function to find homography points from good matches and keypoints
    @staticmethod
    def findHomographyPoints(good_matches, kp1, kp2):
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        return src_pts, dst_pts
    
    def exportPanorama(self):
        # Check if there are at least two images to stitch
        if len(self.images) < 2:
            print("Need at least two images to stitch.")
            return None
        
        print(f"Stitching {len(self.images)} images...")
        
        if self.stitchedImage is None:
            self.stitchedImage = self.images[0]

        for i in range(1, len(self.images)):
            self.stitchedImage = self.stitch(self.stitchedImage, self.images[i])

        return self.stitchedImage

    def stitch(self, img1, img2):
        
        # Step 1: Load images

        # Step 2: Detect keypoints and compute descriptors using SIFT
        sift = cv.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # Step 3: Match descriptors using brute-force matcher
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)        

        # Step 4: Find good matches using Lowe's ratio test
        good_matches = self.findGoodMatches(matches)

        # Step 5: Find homography and warp images if enough matches are found
        if len(good_matches) >= MIN_MATCH_COUNT:
            src_pts, dst_pts = self.findHomographyPoints(good_matches, kp1, kp2)
            H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

        # Step 6: Combine images using the homography
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]
            pts = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
            dst = cv.perspectiveTransform(pts, H)
            # img1 = cv.polylines(img1, [np.int32(dst)], True, (255, 0, 0), 3, cv.LINE_AA)
            print("Homography matrix:\n", H)
            # Warp img1 to img2's perspective
            warped_img2 = cv.warpPerspective(img2, H, (w2+w1, h2))

            # Create a canvas to place the warped image and the original image side by side
            canvas_w = w1 + w2
            canvas_h = max(h1, h2)
            result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            result[:h1, :w1] = img1

            # Create a mask of the warped image and blend it with the original image on the canvas
            gray = cv.cvtColor(warped_img2, cv.COLOR_BGR2GRAY)
            mask = (gray > 0) # Create a mask of the warped image where pixel values are greater than 0
            result[mask] = warped_img2[mask]

            # Crop to bounding box of non-black pixels
            gray_result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
            coords = cv.findNonZero((gray_result > 0).astype(np.uint8))
            x, y, w, h = cv.boundingRect(coords)
            result = result[y:y+h, x:x+w]

            return result
        
if __name__ == "__main__":
    stitcher = PanoramaStitcher()
    cwd = os.getcwd()
    img1 = cv.imread(os.path.join(cwd, "lab04/images/Charlestown1.png")) 
    img2 = cv.imread(os.path.join(cwd, "lab04/images/Charlestown2.jpeg"))
    
    img3 = cv.imread(os.path.join(cwd, "lab04/images/Charlestown3.jpeg"))
    stitcher.addImage(img1)
    stitcher.addImage(img2)
    stitcher.addImage(img3)
    result = stitcher.exportPanorama()
    # print(result.shape)
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title("Panorama Stitching Result")
    plt.axis("off")
    plt.show()
