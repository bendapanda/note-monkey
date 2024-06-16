import numpy as np
from notemonkey.word_segmenter.base_word_segmenter import BaseWordSegmenter

class TrivialWordSegmenter(BaseWordSegmenter):
    def segment(self, image: np.ndarray):
        return []