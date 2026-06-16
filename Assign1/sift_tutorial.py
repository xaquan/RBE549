"""
SIFT, built up step by step
============================

A single, readable SIFT pipeline that stitches together the 7 tutorial steps.
This is the "learn-by-doing" companion to the faithful MATLAB port
(sift_feature.py) - it favours clarity over speed, uses a from-scratch
Gaussian (no cv2.SIFT), and prints/plots what each stage produces.

The seven stages:
    Step 1  load + prepare the image          -> a grid of [0,1] numbers
    Step 2  Gaussian blur (separable 1D)       -> the smoothing building block
    Step 3  Difference of Gaussians (DoG)      -> blob / edge response
    Step 4  scale-space pyramid                -> features at every size
    Step 5  3x3x3 extrema detection            -> raw keypoint candidates (red)
    Step 6  filter: contrast + edge tests      -> stable corners (green, blue)
    Step 7  orientation + 128-D descriptor     -> the matchable fingerprint
    (bonus) match two images with ratio test   -> what SIFT is actually for

Dependencies: numpy, scipy, pillow, matplotlib
Usage:        python sift_tutorial.py path/to/image.png
"""

import sys
import numpy as np
from scipy.signal import convolve2d
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


# ===========================================================================
# Parameters - all the tunable constants in one place
# ===========================================================================
# Step 1 - image preparation
IMAGE_SIZE = 256              # images are resized to IMAGE_SIZE x IMAGE_SIZE

# Step 2 - Gaussian blur
KERNEL_WIDTH_FACTOR = 6       # kernel length ~ KERNEL_WIDTH_FACTOR * sigma
MIN_KERNEL_LENGTH = 3         # never let the kernel get smaller than this

# Steps 3-4 - scale-space pyramid
OCTAVES = 3                   # how many times we halve the image
LEVELS = 3                    # DoG layers per octave
SIGMA0 = np.sqrt(2)           # base blur amount

# Step 6a - contrast rejection
CONTRAST_THRESHOLD = 0.03     # drop keypoints whose DoG value is fainter than this

# Step 6b - edge rejection
EDGE_RATIO = 15.0             # Lowe's r; larger = keep more edge-like points

# Step 7a - orientation
ORIENT_RADIUS = 8             # half-size of the patch used to find the dominant angle
ORIENT_BINS = 36             # direction bins in the orientation histogram (10 deg each)

# Step 7b - descriptor
PATCH_SIZE = 16               # the 16x16 patch sampled around each keypoint
DESCRIPTOR_CELLS = 4          # 4x4 grid of cells
DESCRIPTOR_BINS = 8           # direction bins per cell -> 4*4*8 = 128 numbers
DESCRIPTOR_CLAMP = 0.2        # cap on any single value (lighting robustness)
DESCRIPTOR_DIM = DESCRIPTOR_CELLS * DESCRIPTOR_CELLS * DESCRIPTOR_BINS  # = 128

# Bonus - matching
MATCH_RATIO = 0.7             # Lowe's ratio-test threshold


# ===========================================================================
# Step 2 building block - the Gaussian, done as two cheap 1D passes
# ===========================================================================
def gaussian_1d(length, sigma):
    """A 1D Gaussian bell of the given length, normalized to sum to 1.

    Returned as a ROW (shape (1, N)) so it blurs horizontally; use .T for the
    vertical pass. Reproduces fspecial('gaussian', [1, N], sigma).
    """
    length = max(MIN_KERNEL_LENGTH, int(length))
    half = (length - 1) / 2.0
    x = np.arange(length) - half
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return (g / g.sum()).reshape(1, -1)


def blur(image, sigma):
    """Full 2D Gaussian blur via two 1D passes (separability trick).

    Blurring with a row then a column equals blurring with the full 2D
    kernel, but costs 2*N multiplies per pixel instead of N*N.
    """
    f = gaussian_1d(KERNEL_WIDTH_FACTOR * sigma, sigma)  # kernel width ~ factor*sigma
    out = convolve2d(image, f, mode="same")  # horizontal pass
    out = convolve2d(out, f.T, mode="same")  # vertical pass (f.T = upright bell)
    return out


# ===========================================================================
# Step 1 - load and prepare the image
# ===========================================================================
def load_image(path, size=IMAGE_SIZE):
    """Read -> resize -> grayscale -> scale to [0, 1]. Returns a 2D float array."""
    # img = Image.open(path).resize((size, size)).convert("L")
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (size, size), interpolation=cv.INTER_NEAREST)
    return np.asarray(img, dtype=float) / 255.0


# ===========================================================================
# Steps 3 + 4 - build the scale-space DoG pyramid
# ===========================================================================
def build_dog_pyramid(image, octaves=OCTAVES, levels=LEVELS, sigma0=SIGMA0):
    """Return a list (one entry per octave) of lists of DoG layers.

    Within an octave: blur a bit more each level, subtract neighbours (DoG).
    Across octaves: halve the image and repeat (the pyramid).
    """
    pyramid = []          # pyramid[o] = list of DoG layers at octave o
    base_images = []      # keep each octave's base size for coordinate mapping
    base = image.copy()

    for o in range(octaves):
        base_images.append(base.shape[0])
        # build (levels + 1) blurs so we get `levels` DoG layers
        blurs = []
        for l in range(levels + 1):
            sigma = sigma0 * (2 ** (l / levels))     # increasing blur
            blurs.append(blur(base, sigma))
        dogs = [blurs[i + 1] - blurs[i] for i in range(levels)]   # DoG = subtract
        pyramid.append(dogs)
        base = base[0::2, 0::2]                      # shrink by half -> next octave

    return pyramid, base_images


# ===========================================================================
# Step 5 - find 3x3x3 extrema (raw candidates)
# ===========================================================================
def detect_extrema(pyramid, base_sizes, contrast_threshold=0.0):
    """Scan each interior DoG layer for pixels that are the max/min of their
    27-voxel (3x3x3) neighbourhood. Returns a list of keypoint dicts in
    ORIGINAL-image coordinates.

    contrast_threshold > 0 also applies the Step-6 contrast filter inline.
    """
    keypoints = []
    for o, dogs in enumerate(pyramid):
        scale_factor = 2 ** o                # this octave is shrunk by this much
        # need a layer below and above, so the middle layers are searchable
        for k in range(1, len(dogs) - 1):
            below, mid, above = dogs[k - 1], dogs[k], dogs[k + 1]
            h, w = mid.shape
            for x in range(1, h - 1):
                for y in range(1, w - 1):
                    c = mid[x, y]
                    box = np.concatenate([
                        below[x - 1:x + 2, y - 1:y + 2].ravel(),
                        mid[x - 1:x + 2, y - 1:y + 2].ravel(),
                        above[x - 1:x + 2, y - 1:y + 2].ravel(),
                    ])
                    is_extreme = (c == box.max()) or (c == box.min())
                    if not is_extreme:
                        continue
                    if abs(c) < contrast_threshold:   # Step-6 contrast test
                        continue
                    keypoints.append({
                        "octave": o,
                        "layer": k,
                        "x": x, "y": y,               # within-octave coords
                        "orig_x": x * scale_factor,   # mapped to original
                        "orig_y": y * scale_factor,
                        "value": c,
                        "orientation": None,             # to be filled in Step 7a
                        "descriptor": None               # to be filled in Step 7b
                    })
    return keypoints


# ===========================================================================
# Step 6 - edge rejection (Hessian ratio test)
# ===========================================================================
def reject_edges(keypoints, pyramid, r=EDGE_RATIO):
    """Drop keypoints that sit on an edge (curve strongly one way, flat the
    other). Corners curve both ways and survive.
    """
    R_threshold = (r + 1) ** 2 / r           # = 12.1 when r = 10
    kept = []
    for kp in keypoints:
        dog = pyramid[kp["octave"]][kp["layer"]]
        x, y = kp["x"], kp["y"]
        if x <= 0 or y <= 0 or x >= dog.shape[0] - 1 or y >= dog.shape[1] - 1:
            continue
        # 2x2 Hessian (second derivatives) at the keypoint
        Dxx = dog[x - 1, y] + dog[x + 1, y] - 2 * dog[x, y]
        Dyy = dog[x, y - 1] + dog[x, y + 1] - 2 * dog[x, y]
        Dxy = (dog[x - 1, y - 1] + dog[x + 1, y + 1]
               - dog[x - 1, y + 1] - dog[x + 1, y - 1]) / 4.0
        deter = Dxx * Dyy - Dxy * Dxy
        if deter <= 0:                       # saddle point, not a peak
            continue
        R = (Dxx + Dyy) ** 2 / deter
        if R < R_threshold:                  # corner-like enough -> keep
            kept.append(kp)
    return kept


# ===========================================================================
# Step 7a - dominant orientation of a keypoint
# ===========================================================================
def assign_orientation(image, ox, oy, radius=ORIENT_RADIUS):
    """Build a gradient-direction histogram around (ox, oy) in the
    original image, return the dominant angle in degrees.
    """
    h, w = image.shape
    if ox < radius or oy < radius or ox >= h - radius or oy >= w - radius:
        return None
    patch = image[ox - radius:ox + radius, oy - radius:oy + radius]
    gy, gx = np.gradient(patch)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.degrees(np.arctan2(gy, gx))) % 360

    bin_width = 360 / ORIENT_BINS                # degrees per bin
    hist = np.zeros(ORIENT_BINS)
    for m, a in zip(mag.ravel(), ang.ravel()):
        hist[int(a // bin_width) % ORIENT_BINS] += m
    return np.argmax(hist) * bin_width + bin_width / 2   # center of dominant bin


# ===========================================================================
# Step 7b - the 128-number descriptor
# ===========================================================================
def compute_descriptor(image, ox, oy, orientation=0.0, patch_size=PATCH_SIZE):
    """DESCRIPTOR_CELLS^2 cells x DESCRIPTOR_BINS bins = DESCRIPTOR_DIM numbers,
    normalized + clamped.

    `orientation` is subtracted from every gradient angle so the descriptor
    is measured relative to the keypoint's dominant direction (rotation
    invariance). This is a simplified, axis-aligned sampling - enough to
    learn from; the faithful port rotates the sampling grid itself.
    """
    h, w = image.shape
    half = patch_size // 2
    if ox < half or oy < half or ox >= h - half or oy >= w - half:
        return None

    patch = image[ox - half:ox + half, oy - half:oy + half]
    gy, gx = np.gradient(patch)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.degrees(np.arctan2(gy, gx)) - orientation) % 360   # relative angle

    cell = patch_size // DESCRIPTOR_CELLS         # pixels per cell side
    bin_width = 360 / DESCRIPTOR_BINS             # degrees per direction bin
    descriptor = []
    for i in range(DESCRIPTOR_CELLS):
        for j in range(DESCRIPTOR_CELLS):
            cm = mag[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            ca = ang[i * cell:(i + 1) * cell, j * cell:(j + 1) * cell].ravel()
            hist = np.zeros(DESCRIPTOR_BINS)
            for m, a in zip(cm, ca):
                hist[int(a // bin_width) % DESCRIPTOR_BINS] += m
            descriptor.extend(hist)

    descriptor = np.array(descriptor)
    descriptor = _normalize(descriptor)
    descriptor = np.minimum(descriptor, DESCRIPTOR_CLAMP)  # clamp big values
    descriptor = _normalize(descriptor)                    # renormalize
    return descriptor


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n else v


# ===========================================================================
# Driver - run the whole pipeline and show the red/green/blue plots
# ===========================================================================
def run(image_path, show_plots=True):
    print("Step 1: loading image")
    img = load_image(image_path)
    print(f"        shape {img.shape}, range "
          f"{img.min():.2f}..{img.max():.2f}")

    print("Steps 3-4: building DoG pyramid")
    pyramid, base_sizes = build_dog_pyramid(img)
    print(f"        {len(pyramid)} octaves, "
          f"{len(pyramid[0])} DoG layers each")

    print("Step 5: detecting raw extrema")
    raw = detect_extrema(pyramid, base_sizes, contrast_threshold=0.0)
    print(f"        {len(raw)} raw candidates (red)")

    print("Step 6a: contrast filter")
    contrast = [kp for kp in raw if abs(kp["value"]) >= CONTRAST_THRESHOLD]
    print(f"        {len(contrast)} survive contrast (green)")

    print("Step 6b: edge rejection")
    final = reject_edges(contrast, pyramid)
    print(f"        {len(final)} survive edge test (blue)")

    print("Step 7: orientation + descriptors")
    descriptors, located = [], []
    for kp in final:
        ori = assign_orientation(img, kp["orig_x"], kp["orig_y"])
        if ori is None:
            continue
        d = compute_descriptor(img, kp["orig_x"], kp["orig_y"], ori)
        if d is None:
            continue
        descriptors.append(d)
        located.append(kp)
        kp["orientation"] = ori
        kp["descriptor"] = d
    descriptors = np.array(descriptors)
    print(f"        {len(descriptors)} descriptors, "
          f"each {descriptors.shape[1] if len(descriptors) else 0}-D")

    if show_plots:
        _plot_stages(img, raw, contrast, final, descriptors)

    return descriptors, located


def _plot_stages(img, raw, contrast, final, descriptors):
    """The red -> green -> blue thinning, plus one example descriptor."""
    def xy(kps):
        if not kps:
            return np.empty((0,)), np.empty((0,))
        return (np.array([k["orig_y"] for k in kps]),
                np.array([k["orig_x"] for k in kps]))

    fig, ax = plt.subplots(2, 2, figsize=(11, 11))

    rx, ry = xy(raw)
    ax[0, 0].imshow(img, cmap="gray")
    ax[0, 0].plot(rx, ry, "r+", markersize=4)
    ax[0, 0].set_title(f"Step 5: raw candidates ({len(raw)})")

    cx, cy = xy(contrast)
    ax[0, 1].imshow(img, cmap="gray")
    ax[0, 1].plot(cx, cy, "g+", markersize=5)
    ax[0, 1].set_title(f"Step 6a: contrast-passed ({len(contrast)})")

    fx, fy = xy(final)
    ax[1, 0].imshow(img, cmap="gray")
    ax[1, 0].plot(fx, fy, "b+", markersize=5)
    ax[1, 0].set_title(f"Step 6b: edge-passed ({len(final)})")

    if len(descriptors):
        ax[1, 1].bar(range(DESCRIPTOR_DIM), descriptors[0])
        ax[1, 1].set_title(f"Step 7: one {DESCRIPTOR_DIM}-D descriptor")
        ax[1, 1].set_xlabel(
            f"dimension ({DESCRIPTOR_CELLS*DESCRIPTOR_CELLS} cells "
            f"x {DESCRIPTOR_BINS} dirs)")
    else:
        ax[1, 1].text(0.5, 0.5, "no descriptors", ha="center")

    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.axis("off")
    plt.tight_layout()
    plt.show()

    img_uint8 = (img * 255).astype(np.uint8)
    img_draw = cv.drawKeypoints(img_uint8, [cv.KeyPoint(x=k["orig_y"], y=k["orig_x"], size=1, angle=k["orientation"]) for k in final],
                                None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow('SIFT Keypoints', img_draw)
    cv.waitKey(0)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "test.png"
    descriptors, keypoints = run(path, show_plots=True)
    print(f"\nDone: {len(descriptors)} keypoints described.")