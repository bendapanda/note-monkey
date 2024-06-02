import os
import cv2
import random
import src.preprocessor as preprocessor
from deslant_img import deslant_img
import numpy as np
from segmentation_tests_paper import segment as segment_lines
from segmentation_tests_paper import Line
from sklearn.neighbors import KernelDensity
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from copy import deepcopy
from src.imagehandler import ImageHandler, DeliveryMode



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

def dp_segment_by_whitespace(image: np.ndarray, verbosity:int=0) -> list[np.ndarray]:
    """First goes through every pixel in the top row and attempts to find the shortest
    path to the bottom row that does not touch a black pixel.
    
    Then, analyses the set of lines generated to find the number of seperation points
    (using some sort of clustering algorithm), and makes the cuts"""

    # first, resize the image, and crop it tightly
    image = preprocessor.crop_image_tight(image)
    scaled_down_image = preprocessor.resize_img(image)

    # Perform dp
    cache = np.array([[0 for j in range(scaled_down_image.shape[1])] for i in range(scaled_down_image.shape[0])]).astype(np.float32)
    next_step = cache.copy().astype(int)
    # Now, we need to go bottom up and determine the cost to reach bottom from the nodes.
    # if the node is black, set the cost to infinity
    for i in range(cache.shape[1]):
        if scaled_down_image[-1, i] == 0:
            cache[-1, i] = np.inf

    DIAGONAL_PENALTY = 2
    DOWN_PENALTY = 0
    for y_index in range(cache.shape[0]-2, -1, -1):
        for x_index in range(cache.shape[1]):
            # if the position is a black pixel, we can never make it here
            if scaled_down_image[y_index, x_index] == 0:
                cache[y_index, x_index] = np.inf
                next_step[y_index, x_index] = -1
            else:
                # we have 3 options: down, leftdown, rightdown
                if x_index - 1 < 0:
                    left_score = np.inf
                else:
                    left_score = DIAGONAL_PENALTY + cache[y_index+1, x_index-1]
                if x_index + 1 > cache.shape[1]-1:
                    right_score = np.inf
                else: 
                    right_score = DIAGONAL_PENALTY + cache[y_index+1, x_index+1]
                
                down_score = DOWN_PENALTY + cache[y_index+1, x_index]

                #record the minimum score and therefore what the next steps are
                score = min(left_score, down_score, right_score)
                cache[y_index, x_index] = score
                if score == np.inf:
                    next_step[y_index, x_index] = -1
                elif down_score == score:
                    next_step[y_index, x_index] = x_index
                elif left_score == score:
                    next_step[y_index, x_index] = x_index - 1
                else:
                    next_step[y_index, x_index] = x_index + 1
    
    # Now, we need to construct the paths through the image
    paths = []
    for x_index in range(cache.shape[1]):
        path = []
        current_index = x_index
        if next_step[0, current_index] != -1:
            for y_index in range(cache.shape[0]):
                path.append(current_index)
                current_index = next_step[y_index, current_index]
            paths.append(np.array(path))
    
    paths = np.array(paths)
    # Now, we need to take these paths, and seperate them into clusters.
    labels = dbscan_path_clustering(paths, scaled_down_image.shape[1])

    # for each cluster, find the median line
    clusters = {label: [] for label in labels}
    for i in range(len(paths)):
        clusters[labels[i]].append(paths[i])
    
    median_lines = []
    for path_cluster in clusters.values():
        median_vector = np.median(np.array(path_cluster), axis=0)
        median_lines.append(median_vector.astype(int))
        
    # to test this we need to be able to visualise
    if verbosity>=3:
        # create a random colour per class
        print(f"image shape: {scaled_down_image.shape}")
        print(f"{len(np.unique(labels))} classes found")

        colours = []
        for i in range(len(np.unique(labels))):
            colours.append((random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))
        
        visualised_image = scaled_down_image.copy()
        visualised_image = cv2.cvtColor(visualised_image, cv2.COLOR_GRAY2BGR)
        for start_index in range(len(paths)):
            path = paths[start_index]
            if path is not None:
                for y_value in range(visualised_image.shape[0]):
                    visualised_image[y_value, path[y_value]] = colours[labels[start_index]]
        
        # add the median lines - with a thickness of 3
        for i in range(len(median_lines)):
            for y_index in range(visualised_image.shape[0]):
                if median_lines[i][y_index] < visualised_image.shape[1] -1 :
                    visualised_image[y_index, median_lines[i][y_index]+1] = (0, 0, 255)
                if median_lines[i][y_index] > 0:
                    visualised_image[y_index, median_lines[i][y_index]-1] = (0, 0, 255)
                visualised_image[y_index, median_lines[i][y_index]] = (0, 0, 255)
                #print(y_index, median_lines[i][y_index], visualised_image[y_index, median_lines[i][y_index]])
        
        #visualised_image = preprocessor.resize_img(visualised_image, resize_factor=4)
        cv2.imshow("paths", visualised_image)
        cv2.waitKey(0)

    # finally, use the median paths to create our splits 
    # we crop tight, so in theory, the outsides contain tokens too
    # we also scaled down the image, so we need to work in percentages back in the original image

    # start by scaling the paths in the x-axis
    scaled_paths_x = []
    original_width = image.shape[1]
    scaled_width = scaled_down_image.shape[1]
    for i in range(len(median_lines)):
        scaled_paths_x.append((median_lines[i] * original_width / scaled_width).astype(int))

    # Now the paths are obviously too short. the basic approch is as follows
    desired_paths = []
    original_height = image.shape[0]
    scaled_height = scaled_down_image.shape[0]
    for i in range(len(scaled_paths_x)):
        # create a path of desired length
        desired_length = int(len(scaled_paths_x[i]) * original_height / scaled_height)
        desired_path = [None for j in range(desired_length)]
        # we need to place the original path points at the correct locations
        # At the same time we go through the gaps and just assign them on the lines between points
        old_desired_index = 0
        for scaled_index in range(len(scaled_paths_x[i])):
            desired_index = int(scaled_index * original_height / scaled_height)
            desired_path[desired_index] = scaled_paths_x[i][scaled_index]

            # if have a gap to fill, we should
            if scaled_index > 0:
                num_to_fill = desired_index - old_desired_index
                old_value = desired_path[old_desired_index]
                new_value = desired_path[desired_index]
                for gap_index in range(old_desired_index+1, desired_index):
                    desired_path[gap_index] = int((new_value +old_value)/2)

            old_desired_index = desired_index
        # do we have a problem with no coverage at the end? in some cases yes!
        # to correct, we just set them all to be the same as the last recorded value
        for i in range(old_desired_index+1, original_height):
            desired_path[i] = desired_path[old_desired_index]

        desired_paths.append(desired_path)
    desired_paths = np.array(desired_paths)

    # I need to know what these upscaled paths look like in the context of the original image
    if verbosity >= 4:
        visualised_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for i in range(len(desired_paths)):
            for y_index in range(visualised_image.shape[0]):
                if desired_paths[i][y_index] < visualised_image.shape[1] -1 :
                    visualised_image[y_index, desired_paths[i][y_index]+1] = (0, 0, 255)
                if desired_paths[i][y_index] > 0:
                    visualised_image[y_index, desired_paths[i][y_index]-1] = (0, 0, 255)
                visualised_image[y_index, desired_paths[i][y_index]] = (0, 0, 255)
                #print(y_index, median_lines[i][y_index], visualised_image[y_index, median_lines[i][y_index]])
        
        #visualised_image = preprocessor.resize_img(visualised_image, resize_factor=4)
        cv2.imshow("paths", visualised_image)
        cv2.waitKey(0)




    # Now the paths are scaled, we can chop up our original image
    segments = []
    for index in range(len(desired_paths)):
        last_path = np.zeros(len(desired_paths[index])).astype(int)
        if index > 0:        
            last_path = desired_paths[index-1]
        
        segment_offset = np.min(last_path)
        segment_width = np.max(desired_paths[index]).astype(int) - segment_offset
        segment_image = np.ones((len(desired_paths[index]), segment_width))

        for y_index in range(len(desired_paths[index])):
            segment_start = last_path[y_index]-segment_offset
            segment_end = max(desired_paths[index][y_index]-segment_offset, segment_start)
            segment_image[y_index, segment_start:segment_end] = image[y_index, last_path[y_index]:desired_paths[index][y_index]]
        
        segments.append(segment_image)
    return segments
           

                   



def dbscan_path_clustering(paths:np.array, img_width, eps=0.075, min_samples=3):
    """Clustering algorithm called by dp segment by whitespace
    Given the set of paths, seperates them into clusters and returns the different groups."""
    #TODO hyperparameter adjustment
    #TODO what metric is best?
    #normalise paths to be between 0 and 1
    normalised_paths = paths / img_width
    min_value = 0
    
    cluster_maker = DBSCAN(eps=eps, min_samples=min_samples).fit(normalised_paths)
    return cluster_maker.labels_
    

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
    handler = ImageHandler("line-images")
    handler.image_delivery_mode = DeliveryMode.IN_ORDER
    for i in range(50):
        print(i)
        image = handler.get_new_image()
        image = preprocessor.preprocess_img(image)
      
        #image = preprocessor.resize_img(image, resize_factor=0.5)
        image = preprocessor.remove_inperfections(image)
        image = preprocessor.otsu_thresholding(image)
        #image = preprocessor.hough_transform_rotation(image)
        image = deslant_img(image).img 

        segments = dp_segment_by_whitespace(image, verbosity=3)
        

    #handler.show_image(image)

    #dp_segment_by_whitespace(image)