import numpy as np
import cv2

from line_segmenter.linesegmenter import LineSegmenter
from word_segmenter.wordsegmenter import WordSegmenter

class DPLineSegmenter(LineSegmenter):
    def __init__(self, word_segmenter: WordSegmenter):
        super().__init__(word_segmenter) 

    
