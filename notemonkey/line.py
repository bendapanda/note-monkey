import preprocessor

import numpy as np
import cv2
from word_segmenter.base_word_segmenter import BaseWordSegmenter


class Line():
    """class to represent an individual line
    of handwritten text
    
    NOTE: I am not sure if line should be doing all this processing, feels like that should be handled by the segmenter"""

    def __init__(self, image: np.ndarray, segmenter: BaseWordSegmenter):
        self.image = image
        
        self.segmenter = segmenter
        self.chunks = segmenter.segment(self.image)

      