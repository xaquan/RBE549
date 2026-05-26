# RBE549 - Robotics Computer Vision

This repository is for the **Robotics Computer Vision** course.

## Environment

For this class, I created a Docker environment based on **Ubuntu 24.04**.

### Installed packages

- OpenCV

Error:
```bash
Note that Qt no longer ships fonts. Deploy some (from https://dejavu-fonts.github.io/ for example) or switch to fontconfig.
QFontDatabase: Cannot find font directory /usr/local/lib/python3.12/dist-packages/cv2/qt/fonts.
```

Install font and link to the qt fonts folder
```bash
apt update
apt install -y fonts-dejavu-core fontconfig
mkdir -p /usr/local/lib/python3.12/dist-packages/cv2/qt/fonts
ln -sf /usr/share/fonts/truetype/dejavu/*.ttf \
/usr/local/lib/python3.12/dist-packages/cv2/qt/fonts/
```

## Repository Structure

This repository will contain multiple folders, where each folder is a **lab assignment** for the course.