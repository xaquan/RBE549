# Vision Lab 1 Report

**Course:** RBE549 - Robotics Computer Vision  
**Lab:** Vision-Lab-1  
**Application:** Photo Booth / Camera App  
**Code file:** `cameraApp_part4.py`

## 1. Objective

The objective of this lab was to install and verify an OpenCV Python environment, then develop a webcam-based camera application with common photo booth features. The application captures photos, records video, displays camera overlays, supports live zoom, and applies image processing effects from the OpenCV tutorials.

## 2. Environment

The application was developed in an Ubuntu-based Python environment.

| Component | Version / Tool |
| --- | --- |
| Python | 3.12.3 |
| OpenCV | 4.13.0 |
| Main library | `cv2` |
| Supporting libraries | `os`, `datetime`, `numpy` |

The camera stream is opened with:

```python
cap = cv.VideoCapture(0, cv.CAP_V4L2)
```

The camera format is configured to use MJPG, and the buffer size is set to 1 frame to reduce stream delay.

## 3. Application Controls

| Key / Control | Action |
| --- | --- |
| `c` | Capture and save a photo |
| `v` | Start or stop video recording |
| `e` | Toggle color extraction |
| `r` | Toggle 10 degree rotation |
| `t` | Toggle thresholding |
| `b` | Toggle Gaussian blur and show sigma trackbars |
| `s` | Toggle sharpening |
| `Esc` | Exit the application |
| `+` / `-` on-screen buttons | Zoom in and zoom out |

Captured photos and videos are saved into the `saved_media` directory using timestamped filenames.

## 4. Implemented Features and Outcomes

### Part 2: Basic Camera Application

The application opens a live webcam stream and displays it in an OpenCV window named `xCamera`.

Implemented outcomes:

- Pressing `c` saves the current frame as a JPEG image.
- Pressing `v` starts and stops video recording.
- Pressing `Esc` exits the camera app cleanly.
- Saved photos and videos include a timestamp overlay.
- A live zoom interface is drawn directly on the camera frame.
- Mouse clicks on the on-screen `+` and `-` buttons adjust the zoom level.

The zoom implementation crops the center region of the frame according to the current zoom factor and resizes it back to the original camera resolution. This preserves the display size while creating the visual effect of zooming.

### Part 3: Pixel Operations, ROI, Border, and Blending

Several image arithmetic and ROI operations were added to the live stream.

Implemented outcomes:

- When a photo is captured, the screen briefly flashes white.
- The date/time stamp region is copied from the bottom-right corner and pasted at the top-right corner of the displayed stream.
- A red constant border is added around the live image.
- The OpenCV logo image, loaded from `openCV.png`, is blended into the top-left corner of the stream.

The ROI operation demonstrates direct pixel-region copying, while the logo overlay uses alpha-style blending with `cv.addWeighted`.

### Part 4: Color Spaces, Transformations, Thresholding, and Filtering

The app includes the requested image processing features from the Part 4 tutorials.

Implemented outcomes:

- Pressing `e` extracts a selected color from the stream using HSV color conversion, `cv.inRange`, and a bitwise-AND mask.
- Pressing `r` rotates the camera stream by 10 degrees using `cv.getRotationMatrix2D` and `cv.warpAffine`.
- Pressing `t` thresholds the stream after converting it to grayscale.
- Pressing `b` applies Gaussian blur and enables trackbars for sigma X and sigma Y.
- Pressing `s` sharpens the image using an unsharp masking approach.

Gaussian blur uses adjustable sigma values from 5 to 30. The trackbar values are normalized into that range and then passed into `cv.GaussianBlur`.

The sharpening operation creates a smoothed copy of the frame, subtracts it from the original frame to isolate detail, and blends the detail back into the original image.

## 5. Processing Pipeline

Each camera frame passes through the following processing sequence:

1. Capture a frame from the webcam.
2. Apply zoom if the zoom factor is active.
3. Apply optional color extraction.
4. Apply optional rotation.
5. Apply optional thresholding.
6. Apply optional Gaussian blur.
7. Apply optional sharpening.
8. Copy the frame for saved output.
9. Add date/time stamp to the output frame.
10. Save or record the output frame when requested.
11. Copy the timestamp ROI to the top-right corner of the display frame.
12. Blend the OpenCV logo onto the frame.
13. Add the red border.
14. Draw the zoom controls.
15. Display the final frame.

This order allows the saved image/video output to contain the active visual effects and timestamp, while the display frame also includes interface elements such as the border, logo, copied timestamp ROI, and zoom controls.

## 6. Code Structure

The implementation is organized into small helper functions for individual responsibilities:

| Function | Purpose |
| --- | --- |
| `create_save_folder()` | Creates the output directory if needed |
| `generate_file_path()` | Creates timestamped output paths |
| `add_date_time_to_frame()` | Draws the date/time stamp |
| `save_photo()` | Saves the current frame as an image |
| `start_recording()` / `stop_recording()` | Manages video recording |
| `flash_screen()` | Creates the capture flash effect |
| `apply_zoom()` | Crops and resizes the frame for zoom |
| `copy_datetime_roi()` | Copies the timestamp ROI to the top-right corner |
| `add_opencv_image_to_frame()` | Blends the OpenCV logo into the frame |
| `extract_color()` | Applies HSV masking and bitwise color extraction |
| `rotate_frame()` | Rotates the frame by a specified angle |
| `threshold_frame()` | Applies binary thresholding |
| `apply_gaussian_blur()` | Applies Gaussian filtering |
| `sharpen_frame()` | Applies sharpening through unsharp masking |
| `mouse_callback()` | Handles zoom button clicks |

## 7. Results

The completed application satisfies the requested lab outcomes:

- Live webcam preview is displayed.
- Images can be captured and saved.
- Videos can be recorded and saved.
- Saved media includes a date/time stamp.
- The interface includes live zoom controls.
- The capture action includes a white flash effect.
- The timestamp ROI is copied to a second location.
- A red border is displayed around the stream.
- The OpenCV logo is blended into the camera view.
- Color extraction, rotation, thresholding, Gaussian blur, and sharpening are available through keyboard controls.

## 8. Part 5 Submission Notes

For Part 5, the submission should include:

- `cameraApp_part4.py`, the Python3 source code with explanatory comments.
- `Vision-Lab1-Report.md`, this report document. It can be exported to PDF as `Vision-Lab1-Report.pdf` if required by the submission portal.
- A screen-video recording showing the app while performing the requested key actions: capture, recording, zoom, flash, timestamp ROI copy, border, logo blending, color extraction, rotation, thresholding, Gaussian blur, and sharpening.

## 9. Conclusion

This lab demonstrated the use of OpenCV for live camera capture, media saving, GUI interaction, ROI manipulation, image blending, geometric transformations, color-space masking, thresholding, smoothing, and sharpening. The final camera application combines these features into an interactive photo booth-style tool.
