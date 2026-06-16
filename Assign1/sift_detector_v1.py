import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(CURRENT_DIR, 'images')

class SIFTDetector:
    def __init__(self, img, nOctaveLayers=3, sigma=1.6, contrastThreshold=0.04, edgeThreshold=10):
        self.img = img
        self.gray = img # cv.cvtColor(img, cv.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.nOctaves = self.calculate_nOctaves(img)
        self.nOctaveLayers = nOctaveLayers
        self.k = self.calculate_k(nOctaveLayers)
        self.sigma = sigma
        self.nImagesPerOctave = nOctaveLayers + 3
        self.nDogImagesPerOctave = nOctaveLayers + 2
        self.contrastThreshold = contrastThreshold
        self.edgeThreshold = edgeThreshold

        self.gaussian_pyramid = self.build_gaussian_pyramid()
        self.dog_pyramid = self.build_dog_pyramid(self.gaussian_pyramid)
        interestPoints = self.find_interestPoints()        
        print(f"Found {len(interestPoints)} interest points before localization.")
        self.keypoints = self.clean_keypoints(interestPoints)
        print(f"Found {len(self.keypoints)} keypoints after localization.")

        # img_points = cv.drawKeypoints(self.img, self.keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # cv.imshow('SIFT Keypoints', img_points)

        # plot keypoints on the original image using OpenCV's drawKeypoints function, and display the resulting image in a window titled 'SIFT Keypoints (OpenCV)'
        
        self._plot_keypoints(self.gray, interestPoints, self.keypoints)  # Plot raw candidates and final keypoints using Matplotlib

        sift = cv.SIFT_create()
        kp, des = sift.detectAndCompute(self.gray, None)

        des_img = cv.drawKeypoints(self.gray, kp, self.img, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imshow('SIFT Keypoints (OpenCV)', des_img)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def _plot_keypoints(self, img, raw, final):
        def xy(kps):
            if not kps:
                return np.empty((0,)), np.empty((0,))
            # raw is list of (octave, layer, x, y) tuples; final is list of cv2.KeyPoint
            if isinstance(kps[0], cv.KeyPoint):
                return (np.array([k.pt[0] for k in kps]),
                        np.array([k.pt[1] for k in kps]))
            return (np.array([k[3] for k in kps]),   # col (y)
                    np.array([k[2] for k in kps]))    # row (x)

        fig, ax = plt.subplots(1, 2, figsize=(11, 5))

        final_img = cv.drawKeypoints((img * 255).astype(np.uint8), final,
                                     None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        final_img = cv.cvtColor(final_img, cv.COLOR_BGR2RGB)  # Convert to RGB for Matplotlib

        rx, ry = xy(raw)
        ax[0].imshow(img, cmap="gray")
        ax[0].plot(rx, ry, "r+", markersize=4)
        ax[0].set_title(f"Step 5: raw candidates ({len(raw)})")
        fx, fy = xy(final)
        ax[1].imshow(final_img)
        ax[1].plot(fx, fy, "g+", markersize=4)
        ax[1].set_title(f"Step 6b: edge rejection ({len(final)})")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(2)

    def calculate_nOctaves(self, img):
        min_dim = min(img.shape[0], img.shape[1])
        nOctaves = int(np.floor(np.log2(min_dim))) - 3
        return nOctaves
    
    def calculate_k(self, nOctaveLayers):
        return 2 ** (1 / nOctaveLayers)

    def gaussian_blur(self, img, sigma):
        return cv.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    def calculate_sigma(self, layer):
        res = self.sigma * (self.k ** layer)
        return res
    
    # def build_dog_pyramid(self, gaussian_pyramid):
    #     dog_pyramid = []
    #     for octave_images in gaussian_pyramid:
    #         octave_dogs = []
    #         for i in range(1, len(octave_images)):
    #             dog = cv.subtract(octave_images[i], octave_images[i - 1])
    #             octave_dogs.append(dog)
    #         dog_pyramid.append(octave_dogs)
    #     return dog_pyramid

    #### Optimized by ChatGPT
    def build_dog_pyramid(self, gaussian_pyramid):
        dog_pyramid = []

        for octave_images in gaussian_pyramid:
            octave_dogs = []

            for i in range(1, len(octave_images)):
                dog = octave_images[i] - octave_images[i - 1]
                octave_dogs.append(dog.astype(np.float32))

            dog_pyramid.append(octave_dogs)

        return dog_pyramid

    def compute_gradient(self, cube):
        dx = (cube[1, 1, 2] - cube[1, 1, 0]) / 2.0
        dy = (cube[1, 2, 1] - cube[1, 0, 1]) / 2.0
        ds = (cube[2, 1, 1] - cube[0, 1, 1]) / 2.0
        return np.array([dx, dy, ds])
    
    def compute_hessian(self, cube):
        # cube[s, y, x] where s is the layer index, y is the row index, and x is the column index
        center = cube[1, 1, 1]

        dxx = cube[1, 1, 2] - 2 * center + cube[1, 1, 0]
        dyy = cube[1, 2, 1] - 2 * center + cube[1, 0, 1]
        dss = cube[2, 1, 1] - 2 * center + cube[0, 1, 1]

        dxy = (cube[1, 2, 2] - cube[1, 2, 0] - cube[1, 0, 2] + cube[1, 0, 0])/4.0
        dxs = (cube[2, 1, 2] - cube[2, 1, 0] - cube[0, 1, 2] + cube[0, 1, 0])/4.0
        dys = (cube[2, 2, 1] - cube[2, 0, 1] - cube[0, 2, 1] + cube[0, 0, 1])/4.0

        return np.array([
            [dxx, dxy, dxs], 
            [dxy, dyy, dys], 
            [dxs, dys, dss]
        ])
    
    def compute_offset(self, gradient, hessian):
        return -np.linalg.solve(hessian, gradient)
    
    # def localize_keypoint(self, octave_idx, layer_idx, i, j):
    #     for _ in range(5):  # Max 5 iterations            
    #         subimage1 = self.dog_pyramid[octave_idx][layer_idx - 1][i-1:i+2, j-1:j+2]
    #         subimage2 = self.dog_pyramid[octave_idx][layer_idx][i-1:i+2, j-1:j+2]
    #         subimage3 = self.dog_pyramid[octave_idx][layer_idx + 1][i-1:i+2, j-1:j+2]

    #         cube = np.stack([subimage1, subimage2, subimage3]).astype(np.float32)
    #         gradient = self.compute_gradient(cube)
    #         hessian = self.compute_hessian(cube)

    #         # Compute the offset using the gradient and Hessian. 
    #         # If the Hessian is singular, we cannot localize this keypoint, so we discard it
    #         try:
    #             offset = self.compute_offset(gradient, hessian)
    #         except np.linalg.LinAlgError:
    #             return None  # Singular Hessian, discard this keypoint
            
    #         if np.all(np.abs(offset) < 0.5):
    #             break

    #         j += int(round(offset[0]))
    #         i += int(round(offset[1]))
    #         layer_idx += int(round(offset[2]))

    #         if layer_idx < 1 or layer_idx >= len(self.dog_pyramid[octave_idx]) - 1 or i < 1 or i >= self.dog_pyramid[octave_idx][0].shape[0] - 1 or j < 1 or j >= self.dog_pyramid[octave_idx][0].shape[1] - 1:
    #             return None  # Out of bounds, discard this keypoint
    
    #         # Reject low contrast keypoints
    #         is_low_contrast, contrast = self.reject_low_contrast(cube)
    #         if is_low_contrast:
    #             return None
            
    #         # Reject edge-like keypoints
    #         if self.reject_edge_like(cube):
    #             return None
            
    #         if self.reject_edge_response(hessian):
    #             return None
                        
    #     return (octave_idx, layer_idx, i, j)

    ### Optimized by ChatGPT
    def localize_keypoint(self, octave_idx, layer_idx, i, j):
        max_iter = 5

        for _ in range(max_iter):

            # Check boundary before taking 3x3x3 cube
            if (
                layer_idx < 1 or layer_idx >= len(self.dog_pyramid[octave_idx]) - 1 or
                i < 1 or i >= self.dog_pyramid[octave_idx][0].shape[0] - 1 or
                j < 1 or j >= self.dog_pyramid[octave_idx][0].shape[1] - 1
            ):
                return None

            subimage1 = self.dog_pyramid[octave_idx][layer_idx - 1][i-1:i+2, j-1:j+2]
            subimage2 = self.dog_pyramid[octave_idx][layer_idx][i-1:i+2, j-1:j+2]
            subimage3 = self.dog_pyramid[octave_idx][layer_idx + 1][i-1:i+2, j-1:j+2]

            cube = np.stack([subimage1, subimage2, subimage3]).astype(np.float32)

            gradient = self.compute_gradient(cube)
            hessian = self.compute_hessian(cube)

            try:
                offset = self.compute_offset(gradient, hessian)
            except np.linalg.LinAlgError:
                return None

            dx, dy, ds = offset

            # Good localization: offset is less than half a pixel/layer
            if abs(dx) < 0.5 and abs(dy) < 0.5 and abs(ds) < 0.5:
                break

            # Move to closer sample point
            j += int(round(dx))          # x direction
            i += int(round(dy))          # y direction
            layer_idx += int(round(ds))  # scale direction

        else:
            # Did not converge after max_iter
            return None

        # Recompute cube at final location
        if (
            layer_idx < 1 or layer_idx >= len(self.dog_pyramid[octave_idx]) - 1 or
            i < 1 or i >= self.dog_pyramid[octave_idx][0].shape[0] - 1 or
            j < 1 or j >= self.dog_pyramid[octave_idx][0].shape[1] - 1
        ):
            return None

        subimage1 = self.dog_pyramid[octave_idx][layer_idx - 1][i-1:i+2, j-1:j+2]
        subimage2 = self.dog_pyramid[octave_idx][layer_idx][i-1:i+2, j-1:j+2]
        subimage3 = self.dog_pyramid[octave_idx][layer_idx + 1][i-1:i+2, j-1:j+2]

        cube = np.stack([subimage1, subimage2, subimage3]).astype(np.float32)

        gradient = self.compute_gradient(cube)
        hessian = self.compute_hessian(cube)

        try:
            offset = self.compute_offset(gradient, hessian)
        except np.linalg.LinAlgError:
            return None

        # Lowe interpolated contrast:
        # D(x_hat) = D + 0.5 * gradient.T @ offset
        center_value = cube[1, 1, 1]
        contrast = center_value + 0.5 * np.dot(gradient, offset)

        if abs(contrast) < self.contrastThreshold:
            return None

        # Edge response rejection
        if self.reject_edge_response(hessian):
            return None

        final_x = j + offset[0]
        final_y = i + offset[1]
        final_layer = layer_idx + offset[2]

        # Sigma relative to original image scale
        sigma = self.sigma * (self.k ** final_layer) * (2 ** octave_idx)

        return {
            "octave": octave_idx,
            "layer": final_layer,
            "x": final_x,
            "y": final_y,
            "sigma": sigma,
            "response": abs(contrast)
        }
    
    # def clean_keypoints(self, keypoints):
    #     cleaned_keypoints = []
    #     for octave_idx, layer_idx, i, j in keypoints:
    #         localized_kp = self.localize_keypoint(octave_idx, layer_idx, i, j)
    #         if localized_kp is not None:
    #             keypoint_size = 
    #             cleaned_keypoints.append(cv.KeyPoint(j, i, 1))  # x, y, size
    #     return cleaned_keypoints

    def clean_keypoints(self, keypoints):
        cleaned_keypoints = []

        for octave_idx, layer_idx, i, j in keypoints:
            kp = self.localize_keypoint(octave_idx, layer_idx, i, j)

            if kp is None:
                continue

            scale_factor = 2 ** octave_idx

            x = kp["x"] * scale_factor
            y = kp["y"] * scale_factor
            size = 2 * kp["sigma"]

            cleaned_keypoints.append(
                cv.KeyPoint(
                    x=float(x),
                    y=float(y),
                    size=float(size),
                    response=float(kp["response"]),
                    octave=int(octave_idx)
                )
            )

        return cleaned_keypoints
    
    def reject_low_contrast(self, cube):
        center = cube[1, 1, 1]
        return abs(center) < self.contrastThreshold, abs(center)
    
    def reject_edge_like(self, cube):
        center = cube[1, 1, 1]
        # Compute the second-order derivatives
        dxx = cube[1, 0, 1] - 2 * center + cube[1, 2, 1]
        dyy = cube[0, 1, 1] - 2 * center + cube[2, 1, 1]
        dxy = (cube[0, 0, 1] - cube[0, 2, 1] - cube[2, 0, 1] + cube[2, 2, 1]) / 4.0

        # Compute the trace and determinant of the Hessian matrix
        trace = dxx + dyy
        det = dxx * dyy - dxy ** 2

        # Reject edge-like keypoints
        if det <= 0:
            return True

        # Check the ratio of principal curvatures
        if trace ** 2 / det >= (self.edgeThreshold + 1) ** 2 / self.edgeThreshold:
            return True

        return False
    
    def reject_edge_response(self, hessian):
        dxx = hessian[0, 0]
        dyy = hessian[1, 1]
        dxy = hessian[0, 1]

        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy

        if det <= 0:
            return True

        r = self.edgeThreshold

        if (trace * trace / det) >= ((r + 1) ** 2 / r):
            return True

        return False

    # A pixel is considered a local extremum if it is either greater than all of its 
    # 26 neighbors (a local maximum) or less than all of its 26 neighbors (a local minimum).
    def is_extremum(self, subimage1, subimage2, subimage3):
        center = float(subimage2[1, 1])
        if abs(center) <= self.contrastThreshold:
            return False
        
        cube = np.stack([subimage1, subimage2, subimage3]).astype(np.float32)
        if center > 0:
            return center >= np.max(cube)
        else:
            return center <= np.min(cube)

    
    # For each pixel in the DoG image, we check if it is a local extremum by comparing it to 
    # its 26 neighbors in the current, previous, and next DoG images.
    # def find_interestPoints(self):
    #     keypoints = []
    #     for octave_idx, octave_dogs in enumerate(self.dog_pyramid):
    #         for layer_idx in range(1, len(octave_dogs) - 1):
    #             dog = octave_dogs[layer_idx]
    #             for i in range(1, dog.shape[0] - 1):
    #                 for j in range(1, dog.shape[1] - 1):
    #                     subimage1 = octave_dogs[layer_idx - 1][i-1:i+2, j-1:j+2]
    #                     subimage2 = dog[i-1:i+2, j-1:j+2]
    #                     subimage3 = octave_dogs[layer_idx + 1][i-1:i+2, j-1:j+2]

    #                     if self.is_extremum(subimage1, subimage2, subimage3):
    #                         keypoints.append((octave_idx, layer_idx, i, j))
    #     return keypoints

    ###
    ### Optimized by Claude
    ###
    # def find_interestPoints(self):
    #     keypoints = []
    #     for octave_idx, octave_dogs in enumerate(self.dog_pyramid):
    #         for layer_idx in range(1, len(octave_dogs) - 1):
    #             prev = octave_dogs[layer_idx - 1].astype(np.float32)
    #             curr = octave_dogs[layer_idx].astype(np.float32)
    #             next_ = octave_dogs[layer_idx + 1].astype(np.float32)

    #             # Build 3×3×3 max and min over the cube using dilation/erosion
    #             cube = np.stack([prev, curr, next_])  # (3, H, W)

    #             cube_max = cube.max(axis=0)
    #             cube_min = cube.min(axis=0)

    #             # A pixel is extremum if it equals the global max or min of the cube
    #             is_max = (curr == cube_max)
    #             is_min = (curr == cube_min)
    #             extrema = (is_max | is_min)

    #             # Ignore borders and low contrast
    #             extrema[:1, :] = extrema[-1:, :] = extrema[:, :1] = extrema[:, -1:] = False
    #             extrema[np.abs(curr) <= self.contrastThreshold] = False

    #             ys, xs = np.where(extrema)
    #             for i, j in zip(ys, xs):
    #                 keypoints.append((octave_idx, layer_idx, int(i), int(j)))
    #     return keypoints

    ###
    ### Optimized by ChatGPT
    ###
    def find_interestPoints(self):
        keypoints = []

        for octave_idx, octave_dogs in enumerate(self.dog_pyramid):
            for layer_idx in range(1, len(octave_dogs) - 1):
                prev = octave_dogs[layer_idx - 1].astype(np.float32)
                curr = octave_dogs[layer_idx].astype(np.float32)
                next_ = octave_dogs[layer_idx + 1].astype(np.float32)

                h, w = curr.shape

                # Center pixels, excluding border
                center = curr[1:h-1, 1:w-1]

                neighbors = []

                # Collect 26 neighbors from prev, curr, next scale
                for scale_img, scale_pos in [(prev, -1), (curr, 0), (next_, 1)]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            # skip the center pixel itself
                            if scale_pos == 0 and dy == 0 and dx == 0:
                                continue

                            neighbor = scale_img[
                                1 + dy : h - 1 + dy,
                                1 + dx : w - 1 + dx
                            ]
                            neighbors.append(neighbor)

                neighbors = np.stack(neighbors, axis=0)

                is_max = center > np.max(neighbors, axis=0)
                is_min = center < np.min(neighbors, axis=0)

                extrema = (is_max | is_min)

                # Initial weak response rejection
                extrema[np.abs(center) <= self.contrastThreshold] = False

                ys, xs = np.where(extrema)

                for y, x in zip(ys, xs):
                    # Add +1 because center image excluded border
                    keypoints.append((octave_idx, layer_idx, int(y + 1), int(x + 1)))

        return keypoints


    def build_gaussian_pyramid(self):
        gaussian_pyramid = []
        octave_base = self.gray.copy()

        for octave in range(self.nOctaves):
            octave_images = []
            for layer in range(self.nImagesPerOctave):
                sigma = self.calculate_sigma(layer)
                blurred_img = self.gaussian_blur(octave_base, sigma)
                octave_images.append(blurred_img)

            gaussian_pyramid.append(octave_images)
            # Next octave base is the image at layer nOctaveLayers of the current octave, 
            # downsampled by a factor of 2
            next_octave_base = octave_images[self.nOctaveLayers]
            octave_base = cv.resize(next_octave_base, (next_octave_base.shape[1] // 2, next_octave_base.shape[0] // 2), interpolation=cv.INTER_NEAREST) 
        return gaussian_pyramid


if __name__ == "__main__":

    img = cv.imread(os.path.join(IMAGE_DIR, 'fabio.png'), cv.IMREAD_GRAYSCALE)
    # Resize to 256x256 for faster testing
    img = cv.resize(img, (256, 256), interpolation=cv.INTER_NEAREST)

    SIFTDetector(img, nOctaveLayers=3, sigma=1.6, contrastThreshold=0.01, edgeThreshold=20)