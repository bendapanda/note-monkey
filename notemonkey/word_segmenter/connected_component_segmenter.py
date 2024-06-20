"""
This is my attempt at making a segementer that segments connected components using
opencv rather than dynamic programming which I used before. It will definitately oversegment,
but I am hoping to do some post processing to fix that.
"""
import numpy as np
import cv2

from word_segmenter.base_word_segmenter import BaseWordSegmenter
from model.basemodel import BaseModel
from wordchunk import Chunk
import preprocessor

class ConnectedComponentWordSegmenter(BaseWordSegmenter):
    
    def __init__(self, model: BaseModel):
        super().__init__(model)
    
    def segment(self, line_image: np.ndarray, verbosity: int = 0):
        """chops a line up into the connected components."""

        #first, binarize the line
        binarized_img = preprocessor.preprocess_img(line_image) 
        inverted_img = np.max(binarized_img) - binarized_img

        # now, get all connected components
        num_labels, labeled_img = cv2.connectedComponents(inverted_img)

        if verbosity >= 2:
            # we want to assign each label a colour.
            # Map component labels to hue val
            image_to_show = labeled_img.copy()
            label_hue = np.uint8(179*image_to_show/np.max(image_to_show))
            blank_ch = 255*np.ones_like(label_hue)
            image_to_show= cv2.merge([label_hue, blank_ch, blank_ch])

            # cvt to BGR for display
            image_to_show= cv2.cvtColor(image_to_show, cv2.COLOR_HSV2BGR)

            # set bg label to black
            image_to_show[label_hue==0] = 0

            cv2.imshow('labeled.png', image_to_show)
            cv2.waitKey(0)
        
        # now, we need to take those labels, and segment the image based off of them.
        segmented_sections = []
        for i in range(1, num_labels):
            image_segment = labeled_img == i
            image_segment = 1 - image_segment
            # a big problem with this method is that we don't know what order the connected components are in
            # to solve this we construct an equivelent array containing the x index of the components first pixel
            first_pixel = self.get_first_black_pixel_index(image_segment)
            image_segment = preprocessor.crop_image_tight(image_segment, direction='x') 

            segmented_sections.append((image_segment.astype(np.float32), first_pixel))
        segmented_sections = sorted(segmented_sections, key=lambda x: x[1]) 

        # there are now two things that we need to do: firstly, merge segments that have a high amount of overlap
        self.merge_high_overlap_sections(segmented_sections)

        # then we need to detect spaces in the text by potentially using dpscan with min_neighbours = 2
        # and eps being some factor of the median chunk distance This will cause issues, particularly with overly cursive handwriting
        word_segmented = self.split_into_words(segmented_sections)

        if verbosity >= 4:
            for word in word_segmented:
                for chunk, pixel in word:
                    cv2.imshow(f'{pixel}', chunk.image)
                    cv2.waitKey(0)

        return [[Chunk(section[0], self.model) for section in word] for word in word_segmented]

    def get_first_black_pixel_index(self, image: np.ndarray):
        current_index = image.shape[1]
        for y_index in range(0, image.shape[0], 5):
            for x_index in range(image.shape[1]):
                if image[y_index, x_index] == 0:
                    current_index = min(x_index, current_index)
                    break
        return current_index
    
    def merge_high_overlap_sections(self, sections, merge_modifier=0.9):
        """function that merges sections containing high overlap
        we do this by measuring the percentage that each image overlaps with its neighbours, and if it is high
        enough, then we merge them"""

        old_sections = None
        new_sections = sections
        iterations = 0
        while old_sections != new_sections or iterations < 100:
            iterations += 1
            old_sections = new_sections
            index=0
            while index < len(new_sections)-1:
                overlap_start = max(new_sections[index][1], new_sections[index+1][1])
                overlap_end = min(new_sections[index][1]+new_sections[index][0].shape[1],\
                                  new_sections[index+1][1]+new_sections[index+1][0].shape[1])
                overlap = max(0, overlap_end-overlap_start)
                
                if overlap / new_sections[index][0].shape[1] >= merge_modifier or\
                    overlap / new_sections[index+1][0].shape[1] >= merge_modifier:
                    # we should merge
                    new_img_start = min(new_sections[index][1], new_sections[index+1][1])
                    new_img_stop = max(new_sections[index][1]+new_sections[index][0].shape[1],\
                                  new_sections[index+1][1]+new_sections[index+1][0].shape[1])
                    new_img_width = new_img_stop - new_img_start
                    
                    new_img = np.ones((new_sections[index][0].shape[0], new_img_width), np.uint8)*255
                    print(new_sections[index][0].shape)
                    new_img[:, new_sections[index][1]-new_img_start:\
                            new_sections[index][1]-new_img_start+new_sections[index][0].shape[1]] = new_sections[index][0]
                    new_img[:, new_sections[index+1][1]-new_img_start:\
                            new_sections[index+1][1]-new_img_start+new_sections[index+1][0].shape[1]] = new_sections[index+1][0]

                    new_sections.pop(index+1)
                    new_sections[index] = (new_img, new_img_start)
                else:
                    # if we did not merge two images, hence shortening our list, we need to move along by one
                    index += 1 

    def split_into_words(self, segmented_sections, segment_threshold=2):
        """takes in the segmented sections, along with their starting x indexes, 
        and by considering the distances between their finishing points and the next character's starting point,
        decides whether there is likely to be a space""" 

        distances_between_characters = []
        for index in range(len(segmented_sections)-1):
            this_segment_finish_point = segmented_sections[index][1] + segmented_sections[index][0].shape[1]
            next_segment_start = segmented_sections[index+1][1]
            distances_between_characters.append(max(0, next_segment_start-this_segment_finish_point))
        
        distances_between_characters= np.array(distances_between_characters)
        # for now I'll take the median since it is easy to calculate and should be around about the standard segment gap
        # it might be worth considering other metrics however
        median_gap = np.median(distances_between_characters)
        
        words = []
        current_word = []
        for index in range(len(segmented_sections)-1):
            current_word.append(segmented_sections[index])
            if distances_between_characters[index] > segment_threshold*median_gap:
                # if there is a space
                words.append(current_word)
                current_word = []
        if len(current_word) == 0:
            words.append([segmented_sections[-1]])
        else:
            current_word.append(segmented_sections[-1])
            words.append(current_word)
        
        return words

