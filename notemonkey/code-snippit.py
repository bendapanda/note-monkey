import numpy as np
from line_segmenter.linesegmenter import LineSegmenter
from line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
class CodeSnippit():
    """Class that converts an image into code"""

    def __init__(self, image: np.ndarray, segmenter: LineSegmenter, debug_level:int=0) -> None:
        self.raw_image = image
        self.segmenter = segmenter

        self.debug_level = debug_level

        self.lines = self.segmenter.segment(self.raw_image, self.debug_level)

