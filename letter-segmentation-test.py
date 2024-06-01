import os
import cv2
import preprocessor
from deslant_img import deslant_img
import numpy as np
from segmentation_tests_paper import segment as segment_lines
from segmentation_tests_paper import Line
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from copy import deepcopy



def segment(image, verbosity=0):
    image = preprocessor.crop_image_tight(image)

    image = preprocessor.hough_transform_rotation(image)

    cv2.imshow("rotated", preprocessor.resize_img(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    # IDEA: For preprocessing, try to eliminate rotation in the lines
    image = deslant_img(image).img
    if verbosity >= 3:
        cv2.imshow("deslanted", preprocessor.resize_img(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image = preprocessor.resize_img(image, resize_factor=1)
    thinned_image = preprocessor.thin_text(image)

    cuts = create_clear_cuts(thinned_image, cutoff_factor=0.5)
    pieces = cut_image(cuts, image)
    if verbosity >= 3:
        cv2.imshow("split regions",color_full_splits(thinned_image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(len(pieces))
    if verbosity >= 3:
        for cut in pieces:
            cv2.imshow("cuts", cut)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    for cut in pieces:
        components = get_connected_components(cut)
        for component in components:
            cv2.imshow("connected_component", component)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    

def get_connected_components(image: np.ndarray) -> list[np.ndarray]:
    """Relatively complex, so best to be used with smaller images"""

    # Algorithm idea:
    # go through each pixel
    # if one is black:
    #   remove it and add to new image
    #   look in all 4 directions
    #   recursively remove pixels, adding them to new image
    # repeat
    image = deepcopy(image)
    image = preprocessor.resize_img(image, resize_factor=0.25)

    components = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0:
                new_component = (np.ones(image.shape)*255).astype(np.uint8)
                image, new_component = get_single_connected_component(image, i, j, new_component)
                components.append(new_component)
    return components



def get_single_connected_component(image, i, j, current_connected_image):
    """Given a pixel location, looks at all connected
    pixels of the same colour and returns two images
    the first being the image without any of the connected pixels
    the second being just the connected pixels
    
    ONLY WORKS ON BINARISED IMAGES
    TODO make this function only work on arrays of 0 or 1. seems stupid not to.
    """
    color = image[i, j]

    image[i, j] = 255 - color
    if i > 0:
        # look up
        if image[i-1, j] == color:
            current_connected_image[i-1, j] = color
            
            image, current_connected_image = get_single_connected_component(image, i-1, j, current_connected_image)
    if i < image.shape[0] - 1:
        # look down
        if image[i+1, j] == color:
            current_connected_image[i+1, j] = color
            #image[i+1, j] = 255 - color
            image, current_connected_image = get_single_connected_component(image, i+1, j, current_connected_image)
    if j > 0:
        # look left
        if image[i, j-1] == color:
            current_connected_image[i, j-1] = color
            #image[i, j-1] = 255 - color
            image, current_connected_image = get_single_connected_component(image, i, j-1, current_connected_image)
    if j < image.shape[1]-1:
        # look right
        if image[i, j+1] == color:
            current_connected_image[i, j+1] = color
            #image[i, j+1] = 255 - color
            image, current_connected_image = get_single_connected_component(image, i, j+1, current_connected_image)

    return image, current_connected_image



def cut_image(cuts, image):
    """takes a list of cuts, returns the image cut up
    into those pieces"""
    result = []
    for cut in cuts:
        result.append(image[:, cut[0]:cut[1]])
    return result


def color_full_splits(image):
    new_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(image.shape[1]):
        if np.all(image[:, i] == 255):
            new_image[:, i] = (255, 0, 0)
    return new_image

def create_clear_cuts(image: np.ndarray, cutoff_factor = 1) -> list[tuple[int, int]]:
    """Analyses cuts of clear seperation. looks for spaces?"""
    #TODO: we need to fill in tiny gaps that are close to the mean.
    # After we have the splits, if it is less than the mode split size, just delete it!
    clear_rows = np.zeros(image.shape[1])
    for i in range(image.shape[1]):
        clear_rows[i] = np.all(image[:, i] == 255)


    
    split_points = []
    for i in range(len(clear_rows) - 1):
        if clear_rows[i] != clear_rows[i+1]:
            split_points.append(i)
    split_points = [0] + split_points + [len(split_points) -1]

    # remove trivial splits
    first_gap_split_index = 0 if clear_rows[0] == 1 else 1
    gaps = [(split_points[i], split_points[i+1]) for i in range(first_gap_split_index, len(split_points)-1, 2)]
    section_lengths = np.array([x[1]-x[0] for x in gaps])
    section_lengths = section_lengths.reshape(-1, 1)
    kde = KernelDensity(bandwidth=20).fit(np.sort(section_lengths))
    x = np.linspace(0, np.max(section_lengths), 300)
    y = kde.score_samples(x.reshape(-1, 1))

    mode_index = np.argmax(np.exp(y))
    mode = x[mode_index]

    # remove trivial gap lengths
    valid_gaps = []
    for gap in gaps:
        if gap[1]-gap[0] >= mode * cutoff_factor:
            valid_gaps.append(gap)

    # # convert gaps to the sections we want!
    # sections = [(valid_gaps[i][1], valid_gaps[i+1][0]) for i in range(len(valid_gaps) - 1)]
    # if valid_gaps[0][0] > 0:
    #     sections = [(0, valid_gaps[0][0])] + sections
    # if valid_gaps[-1][1] < len(clear_rows)-1:
    #     sections.append((valid_gaps[-1][1], len(clear_rows)-1))
    


    
    seperated_segments = []
    segment_tuples = []
    previous_part = 0
    for i in range(len(valid_gaps)): # we don't need to do the first and last bits
        split_point = sum(valid_gaps[i]) // 2
        segment_tuples.append((previous_part, split_point))
        seperated_segments.append(image[:, previous_part:split_point])
        previous_part = split_point
    seperated_segments.append(image[:, previous_part:])
    segment_tuples.append((previous_part, image.shape[1]))


    # Finally, clean up trivial splits
    result = []
    for i in range(len(segment_tuples)):
        if not np.all(seperated_segments[i] == 255):
            result.append(segment_tuples[i])

    return result




def main():
    for filename in os.listdir("detection-dataset"):
        image = cv2.imread('detection-dataset/' + filename).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = preprocessor.otsu_thresholding(image)
        lines = segment_lines(image, verbosity=0)
        for line in lines:
            line_image = line.segment_image(image)
            segment(line_image, verbosity=2)

if __name__ == "__main__":
    main()