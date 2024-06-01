import os
import cv2
import preprocessor
from deslant_img import deslant_img
import numpy as np
from segmentation_tests_paper import segment as segment_lines
from segmentation_tests_paper import Line

def segment(image, verbosity=0):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    
    thinned_image = preprocessor.thin_text(image)

    cuts = create_clear_cuts(thinned_image)
    pieces = cut_image(cuts, image)
    if verbosity >= 3:
        for cut in pieces:
            cv2.imshow("cuts", cut)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

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

def create_clear_cuts(image: np.ndarray) -> list[tuple[int, int]]:
    """Analyses cuts of clear seperation. looks for spaces?"""
    clear_rows = np.zeros(image.shape[1])
    for i in range(image.shape[1]):
        clear_rows[i] = np.all(image[:, i] == 255)
    
    seperable_parts = []
    start_index = None
    for i in range(len(clear_rows)):
        if clear_rows[i] == 1 and start_index is None:
            start_index = i
        elif clear_rows[i] == 0  and start_index is not None:
            seperable_parts.append((start_index, i-1))
            start_index = None
    
    seperated_segments = []
    segment_tuples = []
    previous_part = 0
    for i in range(len(seperable_parts)): # we don't need to do the first and last bits
        split_point = sum(seperable_parts[i]) // 2
        segment_tuples.append((previous_part, split_point))
        seperated_segments.append(image[:, previous_part:split_point])
        previous_part = split_point
    seperated_segments.append(image[:, previous_part:])
    segment_tuples.append((previous_part, image.shape[1]))

    # Finally, clean up trivial splits
    result = []
    for i in range(len(seperated_segments)):
        if not np.all(seperated_segments[i] == 255):
            result.append(segment_tuples[i])

    return result




def main():
    for filename in os.listdir("detection-dataset"):
        lines = segment_lines("detection-dataset/" + filename, verbosity=0)
        for line in lines:
            line_image = line.segment_image(preprocessor.otsu_thresholding(cv2.imread('detection-dataset/' + filename).astype(np.uint8)))
            segment(line_image)

if __name__ == "__main__":
    main()