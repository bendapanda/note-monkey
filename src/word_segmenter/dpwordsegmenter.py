import preprocessor as preprocessor
import numpy as np
import random
import cv2
from sklearn.cluster import DBSCAN

from imagehandler import ImageHandler 
from word_segmenter.wordsegmenter import WordSegmenter
from model.basemodel import BaseModel
from wordchunk import Chunk


class DPWordChunkSegmenter(WordSegmenter):
    """
    Class that is responsible for taking lines and segmenting them into cleanly seperable sections
    (seperating non-connected text)
    """
    def __init__(self, model: BaseModel):
        super().__init__(model)

    def segment(self, image: np.ndarray, verbosity:int=0) -> list[np.ndarray]:
        """First goes through every pixel in the top row and attempts to find the shortest
        path to the bottom row that does not touch a black pixel.
        
        Then, analyses the set of lines generated to find the number of seperation points
        (using some sort of clustering algorithm), and makes the cuts"""

        # first, resize the image, and crop it tightly
        image = preprocessor.crop_image_tight(image)
        if image.shape == (1, 0):
            return []
        scaled_down_image = preprocessor.resize_img(image, resize_factor=0.25)

        # Perform dp
        next_step = self._perform_dp(scaled_down_image)
        # Now, we need to construct the paths through the image
        paths = []
        for x_index in range(next_step.shape[1]):
            path = []
            current_index = x_index
            if next_step[0, current_index] != -1:
                for y_index in range(next_step.shape[0]):
                    path.append(current_index)
                    current_index = next_step[y_index, current_index]
                paths.append(np.array(path))
        
        paths = np.array(paths)
        # Now, we need to take these paths, and seperate them into clusters.
        labels = self._dbscan_path_clustering(paths, scaled_down_image.shape[1])

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
            
            visualised_image = (scaled_down_image.copy() * 255).astype(np.uint8)
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

        desired_paths = self._scale_paths(image, scaled_down_image, median_lines, verbosity=verbosity)


        # Now the paths are scaled, we can chop up our original image
        segments = []
        for index in range(len(desired_paths)+1):
            last_path = np.zeros(image.shape[0]).astype(int)
            if index < len(desired_paths):
                current_path = desired_paths[index]
            else:
                current_path = (np.ones(image.shape[0])*image.shape[1]).astype(int)
            if index > 0:        
                last_path = desired_paths[index-1]
            
            segment_offset = np.min(last_path)
            segment_width = np.max(current_path).astype(int) - segment_offset
            segment_image = np.ones((len(current_path), segment_width))*255

            for y_index in range(len(current_path)):
                segment_start = last_path[y_index]-segment_offset
                segment_end = max(current_path[y_index]-segment_offset, segment_start)
                print(last_path[y_index], current_path[index])
                segment_image[y_index, segment_start:segment_end] = image[y_index, last_path[y_index]:current_path[y_index]]

            
            segments.append(Chunk(segment_image, self.model))
        return segments
           

                   
    def _perform_dp(self, image:np.ndarray) -> np.ndarray:
        """Performs DP on the image, and then returns the next_step array"""
        cache = np.array([[0 for j in range(image.shape[1])] for i in range(image.shape[0])]).astype(np.float32)
        next_step = cache.copy().astype(int)
        # Now, we need to go bottom up and determine the cost to reach bottom from the nodes.
        # if the node is black, set the cost to infinity
        for i in range(cache.shape[1]):
            if image[-1, i] == 0:
                cache[-1, i] = np.inf

        DIAGONAL_PENALTY = 2
        DOWN_PENALTY = 0
        for y_index in range(cache.shape[0]-2, -1, -1):
            for x_index in range(cache.shape[1]):
                # if the position is a black pixel, we can never make it here
                if image[y_index, x_index] == 0:
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
        return next_step

    def _scale_paths(self, scaled_image: np.ndarray, image: np.ndarray,
                      median_lines: list[np.ndarray], verbosity:int=0) -> np.ndarray:
        # start by scaling the paths in the x-axis
        scaled_paths_x = []
        scaled_width = scaled_image.shape[1]
        width = image.shape[1]
        for i in range(len(median_lines)):
            scaled_paths_x.append((median_lines[i] * scaled_width / width).astype(int))

        # Now the paths are obviously too short. the basic approch is as follows
        scaled_paths = []
        scaled_height = scaled_image.shape[0]
        height = image.shape[0]
        for i in range(len(scaled_paths_x)):
            # create a path of desired length
            scaled_path_length = int(len(scaled_paths_x[i]) * scaled_height / height)
            scaled_path = [None for j in range(scaled_path_length)]
            # we need to place the original path points at the correct locations
            # At the same time we go through the gaps and just assign them on the lines between points
            old_scaled_path_index = 0
            for pre_yscaled_index in range(len(scaled_paths_x[i])):
                yscaled_index = int(pre_yscaled_index * scaled_height / height)
                scaled_path[yscaled_index] = scaled_paths_x[i][pre_yscaled_index]

                # if have a gap to fill, we should
                if pre_yscaled_index > 0:
                    num_to_fill = yscaled_index - old_scaled_path_index
                    old_value = scaled_path[old_scaled_path_index]
                    new_value = scaled_path[yscaled_index]
                    for gap_index in range(old_scaled_path_index+1, yscaled_index):
                        scaled_path[gap_index] = int((new_value +old_value)/2)

                old_scaled_path_index = yscaled_index
            # do we have a problem with no coverage at the end? in some cases yes!
            # to correct, we just set them all to be the same as the last recorded value
            for i in range(old_scaled_path_index+1, scaled_height):
                scaled_path[i] = scaled_path[old_scaled_path_index]

            scaled_paths.append(scaled_path)
        scaled_paths = np.array(scaled_paths)

        # I need to know what these upscaled paths look like in the context of the original image
        if verbosity >= 4:
            visualised_image = scaled_image.astype(np.uint8) * 255
            visualised_image = cv2.cvtColor(visualised_image, cv2.COLOR_GRAY2BGR)
            for i in range(len(scaled_paths)):
                for y_index in range(visualised_image.shape[0]):
                    if scaled_paths[i][y_index] < visualised_image.shape[1] -1 :
                        visualised_image[y_index, scaled_paths[i][y_index]+1] = (0, 0, 255)
                    if scaled_paths[i][y_index] > 0:
                        visualised_image[y_index, scaled_paths[i][y_index]-1] = (0, 0, 255)
                    visualised_image[y_index, scaled_paths[i][y_index]] = (0, 0, 255)
                    #print(y_index, median_lines[i][y_index], visualised_image[y_index, median_lines[i][y_index]])
            
            #visualised_image = preprocessor.resize_img(visualised_image, resize_factor=4)
            cv2.imshow("paths", visualised_image)
            cv2.waitKey(0)
        return scaled_paths



    def _dbscan_path_clustering(self, paths:np.array, img_width, eps=0.075, min_samples=3):
        """Clustering algorithm called by dp segment by whitespace
        Given the set of paths, seperates them into clusters and returns the different groups."""
        #TODO hyperparameter adjustment
        #TODO what metric is best?
        #normalise paths to be between 0 and 1
        normalised_paths = paths / img_width
        min_value = 0
        
        if len(normalised_paths) <= 0:
            return []
        cluster_maker = DBSCAN(eps=eps, min_samples=min_samples).fit(normalised_paths)
        return cluster_maker.labels_
    

if __name__ == "__main__":
    handler = ImageHandler("lines-dataset/a01/a01-020x")
    
    image = handler.get_new_image()
    image = preprocessor.preprocess_img(image)
    
    #image = preprocessor.resize_img(image, resize_factor=0.5)
    image = preprocessor.remove_inperfections(image)
    #image = preprocessor.hough_transform_rotation(image)
    image = preprocessor.otsu_thresholding(image)/255
    segmenter = DPWordChunkSegmenter()
    segmenter.segment(image)