import cv2 as cv
import numpy as np
import os

CURRENT_DIR = os.getcwd()

# Load images from the specified directory
def load_images():
    img_filename = ["image_over.png", "image_under.png", "image_normal.png"]
    img_list = [cv.imread(os.path.join(CURRENT_DIR, "lab05/images", name)) for name in img_filename]
    
    return img_list

# Merge images using Mertens fusion
# This function takes a list of images and merges them using Mertens fusion, which is a technique for exposure 
# fusion that combines multiple images with different exposures into a single image with better dynamic range.
def merge_images_using_merterns_fusion(img_list):
    merge_mertens = cv.createMergeMertens()
    res = merge_mertens.process(img_list)
    return res

# Save the merged image to the specified path
def save_image(image, filename):
    cv.imwrite(os.path.join(CURRENT_DIR, "lab05/images", filename), image)

def main():
    imgs = load_images()
    hdr_image = merge_images_using_merterns_fusion(imgs)
    hdr_image_8bit = np.clip(hdr_image * 255, 0, 255).astype('uint8')
    save_image(hdr_image_8bit, "hdr_image.png")
    cv.imshow("Merged Image", hdr_image_8bit)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()