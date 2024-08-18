from word_segmenter.base_word_segmenter import BaseWordSegmenter

class LineSegmenter():
    
    def __init__(self, word_segmenter: BaseWordSegmenter, verbosity:int = 0):
        self.verbosity = verbosity
        self.word_segmenter = word_segmenter

    def segment():
        raise NotImplementedError()