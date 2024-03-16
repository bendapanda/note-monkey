"""file that handles preprocessing of images"""
import numpy as np
import cv2


def resize_img(img, resize_factor=0.25):
    y, x = img.shape[:2]
    img = cv2.resize(img, (int(x*resize_factor), int(y*resize_factor)))
    return img


def preprocess_img(img):
    """
    preprocessing with the following steps:
        converting to grayscale
        binarizing
        thinning?
    """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binarized_img = cv2.threshold(
        grayscale_img, 128, 255, cv2.THRESH_BINARY)
    return binarized_img

def thin_text(img):
    """
    thins the text in the given image
    """

    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values    
    thin = np.zeros(img.shape,dtype='uint8')
    img1 = img.copy()
    img1 = blur_image(img1, (10, 10))
    img1 = 255 - img1
    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img1)!=0):
        # Erosion
        erode = cv2.erode(img1,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img1 = erode.copy()

    _, img1 = cv2.threshold(thin, 255/20, 255, cv2.THRESH_BINARY)
    img1 =  255- img1

    return img1

def otsu_thresholding(image):
    """perfroms otsu thresholding on a given image"""
    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use a bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_OTSU,
    )
    return image_result


def blur_image(img, size=(5, 5)):
    new_img = img.copy()
    new_img = 255 - new_img
    new_img = cv2.blur(new_img, size)
    return 255 - new_img

def crop_image_tight(image):
    """Intended for use with a binarised image
    crops to closest fit containing only black pixels"""
    top_dist = 0
    while np.all(image[top_dist] == 255):
        top_dist += 1
    
    bottom_dist = image.shape[0]
    while np.all(image[bottom_dist-1] == 255):
        bottom_dist -= 1
    
    left_dist = 0
    while np.all(image[:, left_dist] == 255):
        left_dist += 1
    
    right_dist = image.shape[1]
    while np.all(image[:, right_dist-1] == 255):
        right_dist -= 1

    return image[top_dist:bottom_dist, left_dist:right_dist]
