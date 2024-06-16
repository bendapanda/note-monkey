import numpy as np
import cv2

from line_segmenter.linesegmenter import LineSegmenter
from notemonkey.word_segmenter.base_word_segmenter import BaseWordSegmenter

class DPLineSegmenter(LineSegmenter):
    def __init__(self, word_segmenter: BaseWordSegmenter):
        super().__init__(word_segmenter) 

    
