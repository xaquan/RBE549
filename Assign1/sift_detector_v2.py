import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, 'images')
IS_DEBUG = True

class SIFTDetector:
    def __init__(self, nOctave=3, nOctaveLayers=3, sigma=1.6, contrastThreshold=0.04, edgeThreshold=10):
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold
        self.nOctave = nOctave
        self.nOctaveLayers = nOctaveLayers
        self.sigma = sigma

        print(f'Initialized SIFTDetector with nOctave={nOctave}, nOctaveLayers={nOctaveLayers}, sigma={sigma}, contrastThreshold={contrastThreshold}, edgeThreshold={edgeThreshold}')

    # Step 0 - initial image preparation
    def image_prepare(self, img, longSide=256):
        """
        This function prepares the input image for SIFT processing by performing the following steps:
        1. Convert the image to float32 and normalize pixel values to the range [0, 1].
        2. Resize the image to have a long side of 256 pixels using nearest neighbor interpolation.
        """
        # Step 1 - initial image preparation
        # Convert to float32 and normalize to [0, 1]
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # Convert to grayscale
        scale = 1.0
        # Resize to have a long side of 256 pixels
        h, w = img.shape
        if max(h, w) > longSide:
            if h > w:
                new_h = longSide
                new_w = int(w * (longSide / h))
            else:
                new_w = longSide
                new_h = int(h * (longSide / w))
            img = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_NEAREST)
            scale = h / new_h  
        
        img = np.asarray(img, dtype=float) / 255.0
        self.resize_scale = scale  
        print(f'Image prepared: original size ({w}, {h}), new size ({img.shape[1]}, {img.shape[0]})')
        return img
    
    # Gaussian blur using a 1D kernel, apply a Gaussian blur to the input image using a separable 1D kernel, 
    # which allows for efficient blurring by first applying the kernel horizontally and then vertically
    def gaussian_1d(self, length, sigma):
        """
        Create a 1D Gaussian kernel.
        Minimum length of the kernel is 3 to ensure it captures the Gaussian shape adequately.
        """
        length = max(3, int(length))
        half = (length - 1)/2
        x = np.arange(length) - half
        g = np.exp(-(x**2)/(2*sigma**2))
        return (g / g.sum()).reshape(1, -1)  # Normalize and reshape to a column vector
    
    # Full 2D Gaussian blur using 2 1D blurs, apply a full 2D Gaussian blur to the input image by 
    # performing two separate convolutions with 1D Gaussian kernels, 
    # first horizontally and then vertically, which is computationally more efficient than using a single 2D kernel
    def blur(self, img, sigma):
        """
        Full 2D Gaussian blur vis 2 1D blurs.
        Kernel width factor is 6
        """
        f = self.gaussian_1d(6*sigma, sigma)
        out = convolve2d(img, f, mode='same')
        out = convolve2d(out, f.T, mode='same')
        return out

    # Step 1-2 build Gaussian pyramid - build DoG pyramid
    def build_dog_pyramid(self, img):
        """
        This function builds a Difference of Gaussians (DoG) pyramid from the input Gaussian pyramid. 
        The DoG pyramid is created by subtracting adjacent blurred images in each octave. 
        The function returns a list of octaves, where each octave is a list of DoG images.
        """
        pyramid = []
        base_sizes = []
        base = img.copy()

        for o in range(self.nOctave):
            base_sizes.append(base.shape[0])

            blurs = []

            for l in range(self.nOctaveLayers + 1):
                sigma = self.sigma * (2 ** (l / self.nOctaveLayers))
                blurred = self.blur(base, sigma)
                blurs.append(blurred)
            dogs = [blurs[i+1] - blurs[i] for i in range(self.nOctaveLayers)]

            pyramid.append(dogs)
            base = base[0::2, 0::2]  # Downsample by a factor of 2 for the next octave

        return pyramid

    # Steps 3-6 - keypoint detection and localization
    def detect_extrema(self, dog_pyramid, contrast_threshold=0.0):
        """
        This function detects keypoints in the DoG pyramid by finding local extrema (maxima and minima) in a 3D neighborhood across scales. 
        It iterates through each octave and each layer of the DoG pyramid, comparing each pixel to its 26 neighbors (8 in the current layer, 9 in the layer above, and 9 in the layer below) 
        to determine if it is a local extremum. The function returns a list of detected keypoints.
        """
        keypoints = []
        for o, dogs in enumerate(dog_pyramid):
            scale_factor = 2 ** o
            for l in range(1, len(dogs)-1): # Avoid the first and last layers of the DoG pyramid, as they do not have neighbors above and below for comparison
                below, mid, above = dogs[l-1], dogs[l], dogs[l+1]
                h, w = mid.shape
                for x in range(1, h-1): # Avoid borders
                    for y in range(1, w-1):
                        c = mid[x, y]
                        box = np.concatenate([
                            below[x-1:x+2, y-1:y+2].ravel(),
                            mid[x-1:x+2, y-1:y+2].ravel(),
                            above[x-1:x+2, y-1:y+2].ravel()
                        ])

                        is_extremum = (c == np.max(box)) or (c == np.min(box))
                        if not is_extremum:
                            continue
                        
                        # Step 6a - contrast rejection
                        if abs(c) < contrast_threshold:
                            continue

                        scale = self.sigma * (2 ** (l / self.nOctaveLayers)) * scale_factor
                        keypoints.append({
                            'octave': o,
                            'layer': l,
                            'x': x,
                            'y': y,
                            'original_x': x * scale_factor,  # Scale back to original image coordinates
                            'original_y': y * scale_factor,  # Scale back to original image coordinates
                            'value': c,
                            'size': scale * 2,  # Keypoint size is typically defined as 2 times the scale at which it was detected
                            'angle': None,  # to be filled in Step 7a
                            'descriptor': None    # to be filled in Step 7b
                        })
        return keypoints
    
    # Step 6b - edge rejection
    def reject_edges(self, dog_pyramid, keypoints):
        """
        This function performs edge rejection on the detected keypoints by analyzing the local 
        curvature of the DoG function at each keypoint.
        """
        r = self.edgeThreshold
        R_threshold = (r + 1) ** 2 / r
        kept_keypoints = []
        for kp in keypoints:
            o = kp['octave']
            l = kp['layer']
            x = kp['x']
            y = kp['y']
            mid = dog_pyramid[o][l]

            if x < 1 or x >= mid.shape[0] - 1 or y < 1 or y >= mid.shape[1] - 1:
                continue  # Skip keypoints too close to the border

            Dyy = mid[x, y+1] + mid[x, y-1] - 2 * mid[x, y]
            Dxx = mid[x+1, y] + mid[x-1, y] - 2 * mid[x, y]
            Dxy = (mid[x+1, y+1] - mid[x+1, y-1] - mid[x-1, y+1] + mid[x-1, y-1]) / 4.0
            
            # Calculate the determinant and trace of the Hessian matrix to perform 
            # edge rejection based on the ratio of principal curvatures.
            det = Dxx * Dyy - Dxy * Dxy
            if det <= 0:
                continue
            R = (Dxx + Dyy) ** 2 / det
            if R < R_threshold:
                kept_keypoints.append(kp)

        return kept_keypoints
    
    # Step 6a - contrast filtering
    def contrast_filter(self, keypoints):
        """
        This function performs contrast filtering on the detected keypoints by removing those whose DoG value is below a specified contrast threshold. 
        It iterates through the list of keypoints and retains only those that have a DoG value greater than the contrast threshold, effectively filtering out low-contrast keypoints that are less likely to be stable and distinctive.
        """
        return [kp for kp in keypoints if abs(kp['value']) >= self.contrastThreshold]


    # Step 7 - orientation assignment     
    def assign_orientation(self, img, ox, oy):
        """
        This function assigns an orientation to a keypoint by computing a histogram of gradient orientations in a local patch around the keypoint. 
        The dominant orientation is determined by finding the peak in the histogram, which represents the most prevalent gradient direction in the patch. 
        The function returns the dominant orientation in degrees.
        """
        radius = 8
        h, w = img.shape
        if ox < radius or ox >= h - radius or oy < radius or oy >= w - radius:
            return None
        # patch = img[ox-radius:ox+radius+1, oy-radius:oy+radius+1]   # [row, col]
        patch = img[oy-radius:oy+radius, ox-radius:ox+radius]   # [col, row]
        # Compute gradients using central differences
        gy, gx = np.gradient(patch)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = np.arctan2(gy, gx) * (180 / np.pi)  # Convert to degrees
        angle[angle < 0] += 360  # Ensure angles are in the range

        # Build histogram of gradient directions
        n_bins = 8
        bin_width = 360 / n_bins                  # 45
        histogram = np.zeros(n_bins)
        for m, a in zip(magnitude.ravel(), angle.ravel()):
            histogram[int(a // bin_width) % n_bins] += m
        dominant_orientation = np.argmax(histogram) * bin_width + bin_width / 2        
        return dominant_orientation 
    
    # Step 8 - keypoint descriptor
    def compute_descriptor(self, img, ox, oy, orientation=0.0):
        """
        Compute a simple descriptor for the keypoint by creating a histogram of gradient orientations in a local patch around the keypoint. 
        The descriptor is a 128-dimensional vector that captures the distribution of gradient orientations in the patch, which can be used for matching keypoints across images.
        """
        radius = 8
        h, w = img.shape

        if ox < radius or ox >= h - radius or oy < radius or oy >= w - radius:
            return None
        
        # patch = img[ox-radius:ox+radius+1, oy-radius:oy+radius+1]   # [row, col]
        patch = img[oy-radius:oy+radius, ox-radius:ox+radius]   # [col, row]

        # Compute gradients using central differences
        gy, gx = np.gradient(patch)
        magnitude = np.sqrt(gx**2 + gy**2)
        angle = (np.degrees(np.arctan2(gy, gx)) - orientation) % 360
        descriptor = []
        # Build histogram of gradient directions
        n_bins = 8
        bin_width = 360 / n_bins  # 8 bins for 45-degree intervals
        for i in range(4):
            for j in range(4):
                sub_magnitude = magnitude[i*4:(i+1)*4, j*4:(j+1)*4]
                sub_angle = angle[i*4:(i+1)*4, j*4:(j+1)*4]
                
                histogram = np.zeros(n_bins)
                for m, a in zip(sub_magnitude.ravel(), sub_angle.ravel()):
                    histogram[int(a // bin_width) % n_bins] += m
                descriptor.extend(histogram)        
        # For simplicity, we will use the histogram as the descriptor. In a full implementation,
        # we would create a more complex descriptor based on the distribution of gradient orientations in a larger patch.
        
        descriptor = np.array(descriptor)
        descriptor = self._normalize_descriptor(descriptor)
        descriptor = np.minimum(descriptor, 0.2)  # Thresholding to reduce the influence of large gradients
        descriptor = self._normalize_descriptor(descriptor)  # Re-normalize after thresholding
        return descriptor
    
    # Main function to detect keypoints and compute descriptors, this function 
    # orchestrates the entire SIFT process by preparing the input image, building the 
    # DoG pyramid, detecting keypoints, performing contrast filtering and edge rejection, 
    # assigning orientations, and computing descriptors for the final set of keypoints. 
    # It returns the list of keypoints and their corresponding descriptors.
    def detectAndCompute(self, image):
        """
        Detect keypoints and compute descriptors for the input image using the SIFT algorithm.
        The function performs the following steps:
        1. Prepare the input image by converting it to grayscale, normalizing pixel values, and resizing it.
        2. Build the DoG pyramid from the prepared image.
        3. Detect keypoints in the DoG pyramid by finding local extrema.
        4. Perform contrast filtering to remove low-contrast keypoints.
        5. Perform edge rejection to remove keypoints that are likely to be on edges.
        6. Assign orientations to the remaining keypoints based on local gradient information.
        7. Compute descriptors for the keypoints based on the distribution of gradient orientations in a local patch around each keypoint.
            The function returns a list of keypoints and their corresponding descriptors.
        """
        self.img = self.image_prepare(image)
        print('Detecting keypoints and computing descriptors...')
        self.dog_pyramid = self.build_dog_pyramid(self.img)
        print('DoG pyramid built.')
        self.raw = self.detect_extrema(self.dog_pyramid, contrast_threshold=0.0)
        print(f'Found {len(self.raw)} keypoints before orientation assignment and descriptor computation.')
        self.contrast = self.contrast_filter(self.raw)
        print(f'Found {len(self.contrast)} keypoints after contrast filtering.')
        self.final = self.reject_edges(self.dog_pyramid, self.contrast)
        print(f'Found {len(self.final)} keypoints after edge rejection.')

        self.descriptors = []
        self.keypoints = []

        for kp in self.final:
            resized_x = kp['original_x'] # Scale back to original image coordinates
            resized_y = kp['original_y']  # Scale back to original image coordinates
            original_x = resized_x * self.resize_scale  # Scale back to original image coordinates
            original_y = resized_y * self.resize_scale  # Scale back to original
            size = kp['size'] * self.resize_scale     # Size of the keypoint based on the scale at which it was detected
            
            orientation = self.assign_orientation(self.img, resized_x, resized_y)
            if orientation is None:
                continue
            descriptor = self.compute_descriptor(self.img, resized_x, resized_y, orientation)  # We will compute the descriptor without rotation for simplicity
            if descriptor is None:
                continue
            kp["angle"] = orientation
            kp["descriptor"] = descriptor

            self.descriptors.append(descriptor)

            keypoint = cv.KeyPoint(x=original_y, 
                                   y=original_x, 
                                   size= size, #self.sigma * (2 ** (l / self.nOctaveLayers)) * 2, 
                                   angle=orientation)
            self.keypoints.append(keypoint)
        self.descriptors = np.array(self.descriptors)

        # if IS_DEBUG:
        #     # Plot keypoints at different stages for visualization
        #     self._plot_keypoints(img, self.raw, self.contrast, self.final, self.keypoints) 

        return self.keypoints, self.descriptors

    # Helper function to normalize the descriptor vector, this function normalizes the input 
    # descriptor vector to have a unit norm, which is important for ensuring that the descriptor 
    # is invariant to changes in illumination and contrast.
    def _normalize_descriptor(self, descriptor):
        norm = np.linalg.norm(descriptor)
        return descriptor / norm if norm > 0 else descriptor
    
    # Plot keypoints at different stages for visualization, this function visualizes 
    # the keypoints detected at various stages of the SIFT algorithm by plotting them on 
    # the original image using Matplotlib.
    # def _plot_keypoints(self, img, raw, contrast, final, complete_keypoints):
    def _plot_keypoints(self):
        def xy(kps):
                if not kps:
                    return np.empty((0,)), np.empty((0,))
                return (np.array([k["original_y"] for k in kps]),
                        np.array([k["original_x"] for k in kps]))
    
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))

        
        final_img = cv.drawKeypoints((self.img * 255).astype(np.uint8), [cv.KeyPoint(x=k["original_y"], y=k["original_x"], size=k["size"], angle=k["angle"]) for k in self.final],
                                        None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

        rx, ry = xy(self.raw)
        ax[0, 0].imshow(self.img, cmap="gray")
        ax[0, 0].plot(rx, ry, "r+", markersize=4)
        ax[0, 0].set_title(f"Step 5: raw candidates ({len(self.raw)})")
        cx, cy = xy(self.contrast)
        ax[0, 1].imshow(self.img, cmap="gray")
        ax[0, 1].plot(cx, cy, "b+", markersize=4)
        ax[0, 1].set_title(f"Step 6a: contrast filtering ({len(self.contrast)})")
        fx, fy = xy(self.final)
        ax[1, 0].imshow(self.img, cmap="gray")
        ax[1, 0].plot(fx, fy, "g+", markersize=4)
        ax[1, 0].set_title(f"Step 6b: edge rejection ({len(self.final)})")
        ox, oy = xy(self.final)
        ax[1, 1].imshow(final_img, cmap="gray")
        # ax[1, 1].plot(ox, oy, "m+", markersize=4)
        ax[1, 1].set_title(f"Step 7: orientation assignment ({len(self.keypoints)})")
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)
        
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

if __name__ == "__main__":

    img = cv.imread(os.path.join(IMAGE_DIR, 'lenna.png'))
    img = cv.resize(img, (int(img.shape[1]*0.67), int(img.shape[0]*0.67)), interpolation=cv.INTER_NEAREST)

    img2 = cv.imread(os.path.join(IMAGE_DIR, 'lenna.png'))
    img2 = cv.resize(img2, (int(img2.shape[1]*0.67), int(img2.shape[0]*0.67)), interpolation=cv.INTER_NEAREST)
    
    sift = SIFTDetector(nOctave=6, nOctaveLayers=3, sigma=1.6, contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(img)

    sift._plot_keypoints()  # Plot keypoints at different stages for visualization

    # Visualize keypoints and their orientations
    img_draw_keypoint = cv.drawKeypoints(img, kp1, None)
    img_draw_orientation = cv.drawKeypoints(img, kp1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypointsAndOrientationImg = np.hstack((img_draw_keypoint, img_draw_orientation))

    # Visualize matches between keypoints (for demonstration, we will match the descriptors to themselves)
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=False)
    kp2, des2 = sift.detectAndCompute(img2)
    matches = bf.knnMatch(des2.astype(np.float32), des1.astype(np.float32), k=2)
    drawedMatchesImg = draw_matches(img2, kp2, img, kp1, matches)

    # plot keypointsAndOrientationImg and drawedMatchesImg vertically in matplotlib
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(cv.cvtColor(keypointsAndOrientationImg, cv.COLOR_BGR2RGB))
    plt.title('Keypoints and Orientation')
    plt.axis('off')
    plt.subplot(2, 1, 2)
    plt.imshow(cv.cvtColor(drawedMatchesImg, cv.COLOR_BGR2RGB))
    plt.title('Matches')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

