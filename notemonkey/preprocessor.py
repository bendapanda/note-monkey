"""file that handles preprocessing of images"""
import numpy as np
import cv2


def resize_img(img: np.ndarray, resize_factor=0.25) -> np.ndarray:
    
    y, x = img.shape[:2]
    resized_x = int(x*resize_factor)
    resized_y = int(y*resize_factor)
    if resized_x <= 0 or resized_y <= 0:
        raise ValueError("resize values too small to be handled")
    img = cv2.resize(img, (resized_x, resized_y))
    return img


def preprocess_img(img: np.ndarray):
    """
    preprocessing with the following steps:
        converting to grayscale
        binarizing
    """
    if img.shape[-1] == 0:
        raise ValueError('zero dimensional images cannot be preprocessed')
        
    if len(img.shape) == 2:
        grayscale_img = img.copy()
    elif len(img.shape) == 3:
        grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError('image can only be 2 or 3 channel')
    
   
    binarized_img = otsu_thresholding(grayscale_img)
    return binarized_img

def dialate_by_pixel_density(image: np.ndarray, max_iterations = 5, kernal_size=5) -> np.ndarray:
    """I want to be able to dialate images more on rows where they have a higher pixel density
    """
    # first, make sure it is greyscale, else throw a valueerror
    if len(image.shape) != 2:
        raise ValueError("input image should be in greyscale")
    if len(np.unique(image)) > 2:
        raise ValueError("input image must be binarized")
    if image.shape[1] < kernal_size: 
        raise ValueError("image width must be at least as big as the kernal")
    dialated_image = image.copy()
    dialated_image = np.max(dialated_image) - dialated_image
    print(dialated_image) 
    # first, get pixel counts
    row_counts = np.sum(image, axis=1)
    row_counts = row_counts / np.max(row_counts)

    # now dialate each row proportionally
    kernel = np.ones((1,kernal_size), np.uint8)
    for index in range(image.shape[0]):
        row = dialated_image[index:index+1]
        dialeted_row = cv2.dilate(row, kernel, iterations=int(max_iterations*row_counts[index]))
        dialated_image[index:index+1] = dialeted_row
    return dialated_image


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

def hough_transform_rotation(image):
    """finds the most likely angle of the lines in the image, and will rotate the image so the line is level"""
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    rotation_angle = np.median(angles)
    
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated_image



def blur_image(img, size=(5, 5)):
    new_img = img.copy()
    new_img = 255 - new_img
    new_img = cv2.blur(new_img, size)
    return 255 - new_img

def crop_image_tight(image, direction='both'):
    """Intended for use with a binarised image
    crops to closest fit containing only black pixels"""
    
    if type(image) != np.ndarray:
        raise ValueError('cannot crop something which is not an image')
    if image.shape[0] == 0 or image.shape[1] == 0:
        return image

    white_value = 1 if 1 in np.unique(image) else 255
    if not np.all(image == white_value):

        top_dist = 0
        bottom_dist = image.shape[0]
        if direction == 'both' or direction == 'y':
            while np.all(image[top_dist] == white_value):
                top_dist += 1
            
            while np.all(image[bottom_dist-1] == white_value):
                bottom_dist -= 1
            
        left_dist = 0
        right_dist = image.shape[1]
        if direction == 'both' or direction == 'x':
            while np.all(image[:, left_dist] == white_value):
                left_dist += 1
            
            while np.all(image[:, right_dist-1] == white_value):
                right_dist -= 1
        return image[top_dist:bottom_dist, left_dist:right_dist]
    else:
        return np.array([[]])

def remove_inperfections(image: np.ndarray) -> np.ndarray:
    """Takes an image as inputs and tries to smooth out small imperfections caused by pen strokes, etc"""
    # invert the image
    inverted_image = cv2.bitwise_not(image)

    # add blur to the image to try and soften jagged edges
    blurred_image = cv2.GaussianBlur(inverted_image, (5, 5), 0)


    # Apply morphological closing operation (dilation followed by erosion)
    closing_kernel = np.ones((3, 3), np.uint8)
    closing_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, closing_kernel)

    # apply smoothing
    smoothed_image = cv2.morphologyEx(closing_image, cv2.MORPH_OPEN, closing_kernel)

    final_image = cv2.bitwise_not(smoothed_image)
    return final_image

if __name__ == "__main__":
    filepath = "detection-dataset/20240128_090230.jpg"
    image = cv2.imread(filepath)
    image = preprocess_img(image)
    cv2.imshow("before", image)
    cv2.waitKey(0)
    
    # remove imperfections
    new_image = remove_inperfections(image)
    cv2.imshow("after", new_image)
    cv2.waitKey(0)

