"""
Find a book on a table using local feature matching.

Pipeline:
  1. Detect + describe keypoints in both images with SIFT and SURF.
  2. Match descriptors with Brute-Force and FLANN.
  3. Filter matches with Lowe's ratio test.
  4. Estimate a homography (RANSAC) and draw the book's outline on the table.

Notes on SURF:
  SURF is patented ("nonfree"). It lives in cv2.xfeatures2d and is only present
  if you installed an opencv-contrib-python build compiled with
  OPENCV_ENABLE_NONFREE=ON. The stock pip wheels are NOT compiled that way, so
  cv2.xfeatures2d.SURF_create() will raise. The code below detects this and
  skips SURF gracefully (or you can swap in ORB/AKAZE -- see USE_SURF_FALLBACK).
"""
import os

import cv2
import numpy as np

# ---------------------------------------------------------------- config
QUERY_PATH = "lab03/book.png"      # the object to find
TRAIN_PATH = "lab03/book_in_scene.png"     # the scene containing the object
RATIO      = 0.75            # Lowe's ratio-test threshold
MIN_GOOD   = 10             # min good matches needed to attempt homography
USE_SURF_FALLBACK = True    # if SURF unavailable, use AKAZE instead of skipping


# ---------------------------------------------------------------- detectors
def build_detectors():
    """Return a dict of {name: detector}. SURF added only if available."""
    detectors = {"SIFT": cv2.SIFT_create()}

    try:
        # hessianThreshold ~ 400 is a common default; higher = fewer points
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        detectors["SURF"] = surf
    except (cv2.error, AttributeError) as e:
        print(f"[!] SURF unavailable ({type(e).__name__}). "
              f"Your OpenCV was not built with the nonfree module.")
        if USE_SURF_FALLBACK:
            print("[*] Falling back to AKAZE for the second detector.")
            detectors["AKAZE"] = cv2.AKAZE_create()

    return detectors


# ---------------------------------------------------------------- matchers
def bf_match(desc1, desc2, norm):
    """Brute-Force KNN match (k=2) for the ratio test. crossCheck must be off."""
    bf = cv2.BFMatcher(norm, crossCheck=False)
    return bf.knnMatch(desc1, desc2, k=2)


def flann_match(desc1, desc2, binary):
    """FLANN KNN match. KD-tree for float descriptors (SIFT/SURF),
    LSH for binary descriptors (ORB/AKAZE)."""
    if binary:
        index_params = dict(algorithm=6,        # FLANN_INDEX_LSH
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
    else:
        index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    return flann.knnMatch(desc1, desc2, k=2)


def ratio_test(knn_matches, ratio=RATIO):
    """Lowe's ratio test. Guards against degenerate pairs (<2 neighbours)."""
    good = []
    for m_n in knn_matches:
        if len(m_n) < 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


# ---------------------------------------------------------------- localization
def locate_object(kp1, kp2, good, img_query, img_train):
    """Estimate homography from good matches and draw the projected box."""
    if len(good) < MIN_GOOD:
        print(f"    Not enough matches: {len(good)}/{MIN_GOOD}")
        return None, None

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("    Homography estimation failed.")
        return None, None

    h, w = img_query.shape[:2]
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(corners, H)

    scene = img_train.copy()
    scene = cv2.polylines(scene, [np.int32(projected)], True, (0, 255, 0), 3, cv2.LINE_AA)
    inliers = int(mask.sum())
    print(f"    Homography OK -- {inliers}/{len(good)} inliers")
    return scene, mask


# ---------------------------------------------------------------- main
def main():
    root = os.getcwd()
    img_query = cv2.imread(os.path.join(root, QUERY_PATH))
    img_train = cv2.imread(os.path.join(root, TRAIN_PATH))
    if img_query is None or img_train is None:
        raise FileNotFoundError(f"Could not load '{QUERY_PATH}' and/or '{TRAIN_PATH}'")

    gray_q = cv2.cvtColor(img_query, cv2.COLOR_BGR2GRAY)
    gray_t = cv2.cvtColor(img_train, cv2.COLOR_BGR2GRAY)

    detectors = build_detectors()

    for name, det in detectors.items():
        print(f"\n=== {name} ===")
        kp1, des1 = det.detectAndCompute(gray_q, None)
        kp2, des2 = det.detectAndCompute(gray_t, None)
        print(f"  keypoints: query={len(kp1)}  train={len(kp2)}")

        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            print("  Too few descriptors; skipping.")
            continue

        # Binary descriptors (AKAZE/ORB) need Hamming norm + LSH FLANN.
        binary = des1.dtype == np.uint8
        norm = cv2.NORM_HAMMING if binary else cv2.NORM_L2

        for matcher_name, knn in [
            ("Brute-Force", bf_match(des1, des2, norm)),
            ("FLANN",       flann_match(des1, des2, binary)),
        ]:
            good = ratio_test(knn)
            print(f"  [{matcher_name}] good matches: {len(good)}")

            scene, _ = locate_object(kp1, kp2, good, gray_q, img_train)
            if scene is not None:
                out = f"result_{name}_{matcher_name.replace('-', '')}.jpg"
                cv2.imwrite(out, scene)
                print(f"    wrote {out}")

            # side-by-side match visualization
            vis = cv2.drawMatches(
                img_query, kp1, img_train, kp2, good, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(f"matches_{name}_{matcher_name.replace('-', '')}.jpg", vis)


if __name__ == "__main__":
    main()