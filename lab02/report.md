# Vision Lab 2 Report
**RBE 549 — Computer Vision**
**Author:** Quan Trinh

---

## Overview

This lab explores image gradient computation, edge detection, geometric transformations, and feature extraction. The work is organized into four parts, each building on the previous, and integrates new capabilities into the PhotoBooth/Camera application from Lab 1.

---

## Part 1 — Image Gradients and Canny Edge Detection (lab02_Part1.py)

### Background

The Sobel operator computes image gradients by convolving the image with derivative kernels in the horizontal (X) and vertical (Y) directions. Gradients highlight intensity transitions, which correspond to edges. The Canny edge detector is a multi-stage algorithm: it applies Gaussian smoothing, computes gradients, performs non-maximum suppression, and applies hysteresis thresholding to produce thin, precise edges.

### Implementation

New features were added to the camera app from Lab 1:

**a) Sobel X (`g` then `x`):**
Pressing `g` puts the app into a mode that waits for a direction key. Pressing `x` creates a trackbar labeled "Sobel X" on the main window. The trackbar value is mapped to odd kernel sizes (value × 2 + 1), ensuring valid Sobel kernel dimensions. `cv.Sobel()` is applied with `dx=1, dy=0`.

**b) Sobel Y (`g` then `y`):**
Same flow as above but pressing `y` creates the "Sobel Y" trackbar. `cv.Sobel()` is applied with `dx=0, dy=1`.

**c) Canny Edge Detection (`d`):**
Pressing `d` creates two trackbars — "Canny Threshold 1" and "Canny Threshold 2" — each ranging from 1 to 5000. The frame is converted to grayscale, `cv.Canny()` is applied with the live threshold values, and the result is converted back to BGR for display consistency.

```python
def apply_canny_edge_detection(frame):
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray_frame, canny_threshold_1, canny_threshold_2)
    frame[:] = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
```

**Key bindings summary:**

| Key | Action |
|-----|--------|
| `g` → `x` | Sobel X with kernel size trackbar |
| `g` → `y` | Sobel Y with kernel size trackbar |
| `d` | Canny edge detection with two threshold trackbars |
| `v` | Toggle video recording |
| `c` | Capture photo |
| `b` | Toggle Gaussian blur with sigma trackbars |
| `s` | Toggle image sharpening |
| `t` | Toggle binary threshold |
| `e` | Toggle color extraction (pink) |
| `r` | Toggle 10° rotation |
| `ESC` | Exit |

---

## Part 2 — Custom Sobel and Laplacian Operators (lab02_Part2.py)

### Background

The Sobel operator approximates the image gradient using finite differences with a weighted 3×3 kernel. The Laplacian operator computes the second derivative of image intensity using a discrete approximation, making it sensitive to regions of rapid intensity change (edges) in all directions simultaneously.

### Implementation

Both operators were implemented **without** using `cv.Sobel()` or `cv.Laplacian()`. Instead, a general 2D convolution function was written manually using NumPy:

```python
def _convolve3x3(gray_frame, kernel):
    h, w = gray_frame.shape
    out = np.zeros((h, w), dtype=np.float32)
    for ki in range(3):
        for kj in range(3):
            if kernel[ki, kj] != 0:
                out[1:h-1, 1:w-1] += kernel[ki, kj] * gray_frame[ki:h-2+ki, kj:w-2+kj]
    return out
```

This vectorized approach shifts the input array and accumulates weighted contributions for each non-zero kernel element, avoiding nested pixel loops while remaining pure NumPy.

**Custom Sobel:**
```python
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
```
The gradient magnitude `G = sqrt(Gx² + Gy²)` is clipped to [0, 255] and returned as a displayable image.

**Custom Laplacian:**
```python
laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
```
The 4-connected Laplacian kernel computes the sum of second derivatives and is sensitive to edges in all directions.

**Four-window display (pressing `4`):**
Pressing `4` captures the current camera frame and opens a Matplotlib figure arranged in a 2×2 grid showing:
- Original
- Laplacian
- Sobel X
- Sobel Y

```python
if key == ord('4'):
    plot_filters(frame)
```

### Observations

- The Laplacian responds to edges in all directions but is sensitive to noise, producing thin double-edge responses.
- Sobel X highlights vertical edges (horizontal intensity change), while Sobel Y highlights horizontal edges (vertical intensity change).
- The custom implementations produce output visually equivalent to OpenCV's built-in operators.

---

## Part 3 — Geometric Transformations (lab02_Part3.py / lab02_Part3-4.py)

### Background

Geometric transformations modify the spatial arrangement of pixels. Affine transformations (rotation, scaling, shear) preserve parallelism. Perspective transformations model the effect of viewing a planar surface from a different angle, useful for correcting projective distortion.

### Implementation

All five transformations were applied to `UnityHall.png`:

**1. Rotation by 10 degrees**
`cv.getRotationMatrix2D()` generates the 2×3 affine matrix. The output canvas size is computed to contain the full rotated image without clipping:
```python
new_w = int(h * sin + w * cos)
new_h = int(h * cos + w * sin)
```

**2. Scale Up (+20%)**
`cv.resize()` with `fx=1.2, fy=1.2` and `INTER_LINEAR` interpolation (appropriate for upsampling).

**3. Scale Down (−20%)**
`cv.resize()` with `fx=0.8, fy=0.8` and `INTER_AREA` interpolation (appropriate for downsampling, reduces aliasing).

**4. Affine Transformation**
Three point correspondences are defined, mapping `pts1 → pts2`. `cv.getAffineTransform()` computes the 2×3 matrix. The output bounding box is computed by transforming all four corners and adjusting the translation so no coordinates go negative.

**5. Perspective Transformation**
Four point correspondences simulate a slight vertical distortion at the bottom of the image. `cv.getPerspectiveTransform()` produces a 3×3 homography matrix. The output canvas is sized by transforming all corners, and a translation matrix `T` is prepended to shift the result into positive coordinates:
```python
T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
M = T @ M
```

All images are normalized to the same canvas size (white padding) for side-by-side display.

---

## Part 4 — Harris Corner Detection and SIFT (lab02_Part3-4.py)

### Background

**Harris Corner Detection** measures the change in image intensity in all directions by analyzing the eigenvalues of the second-moment matrix (structure tensor). A corner is detected when both eigenvalues are large, indicating high variation in all directions. The corner response is:

```
R = det(M) - k * trace(M)²
```

**SIFT (Scale-Invariant Feature Transform)** detects keypoints that are stable across scale and rotation changes by finding extrema in a Difference-of-Gaussians (DoG) scale-space pyramid. Each keypoint receives a descriptor based on local gradient histograms, making it robust to illumination and viewpoint changes.

### Implementation

Both algorithms are applied to all six images (Original, Rotated, Scale Up, Scale Down, Affine, Perspective):

**Harris (red markers):**
```python
gray = cv.GaussianBlur(gray, (5, 5), 1.5)        # suppress tree texture before detection
dst = cv.cornerHarris(gray, blockSize=5, ksize=7, k=0.04)
dst_dilated = cv.dilate(dst, None)
image[dst_dilated > 0.08 * dst_dilated.max()] = [0, 0, 255]
```
Parameters held constant across all images: Gaussian pre-blur (5×5, σ=1.5), block size = 5, Sobel aperture = 7, k = 0.04, threshold = 8% of max response.

**SIFT (green markers):**
```python
sift = cv.SIFT_create(
    nfeatures=200, nOctaveLayers=3,
    contrastThreshold=0.05, edgeThreshold=10, sigma=1.6
)
keypoints = sift.detect(gray, None)
transformed = cv.drawKeypoints(image, keypoints, None, (0, 255, 0))
```
Parameters held constant across all images. `nfeatures=200` caps total detections; `contrastThreshold=0.05` and `edgeThreshold=10` are tuned to recover structural building corners while limiting tree clutter.

Results are displayed as two combined 2×3 grids — one for Harris detections and one for SIFT detections — with captions added to each panel.

### Observations

- **Both detectors** concentrated on the tree region initially, since foliage produces many high-frequency intensity transitions that both Harris and SIFT respond to strongly. Major structural corners — the roofline of the building and the wall at the lower boundary — were missed in the initial run.
- **Tuning rationale:** The Harris block size was increased from 2 to 5 and a Gaussian pre-blur was added to average out leaf-scale texture and focus the corner response on larger, structural junctions. The threshold was raised from 2% to 8% of the maximum response to keep only the strongest corners. For SIFT, `contrastThreshold=0.15` was over-suppressing building corners along with weak blobs; lowering it to 0.05 recovers those detections. `edgeThreshold` was raised from 4 to 10 (default) because building wall corners sit near strong edges and were being discarded at the stricter setting.
- **Harris** is still not scale-invariant: the scaled-down image loses some corners detected in the original, while scaled-up spreads the same corners over a larger area.
- **SIFT** remains more consistent across the scale variants due to its DoG scale-space construction, though the 200-feature cap means weaker keypoints are dropped in complex scenes.
- **Perspective transformation** warps straight architectural lines, generating apparent corners at the distorted boundaries in both Harris and SIFT outputs.
- **Affine transformation** preserves parallelism, so detections closely mirror the original image.

---

## Summary

| Part | File | Key Features |
|------|------|-------------|
| 1 | lab02_Part1.py | Sobel X/Y with trackbar, Canny with dual threshold trackbars |
| 2 | lab02_Part2.py | Custom Sobel and Laplacian via NumPy convolution, 4-panel display on `4` |
| 3 | lab02_Part3.py | Rotation, scale ±20%, affine, perspective on UnityHall.png |
| 4 | lab02_Part3-4.py | Harris (red) + SIFT (green) on all 6 transformed images |

All parts use `opencv-contrib-python` for SIFT support and are implemented in Python 3 with OpenCV 4.x and NumPy.
