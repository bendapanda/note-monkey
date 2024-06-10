import numpy as np
import pytest
from src.word_segmenter.dpwordsegmenter import DPWordChunkSegmenter

class TestDPWordChunkSegmenter():
    def __init__(self):
        self.segmenter = DPWordChunkSegmenter() 
    
    def test_on_1by1(self):
        """we expect to get no characters here"""
        one_by_one = np.array([[255]])
        output = self.segmenter.segment(one_by_one)
        assert output == []

    def test_on_empty(self):
        empty = np.array([[]])
        output = self.segmenter.segment(empty)
        assert output == []

    def test_on_image_no_text(self):
        input = np.ones((100, 300))*255
        output = self.segmenter.segment(input)
        assert output == []

    def test_on_all_black(self):
        input = np.zeros((100,300))
        output = self.segmenter.segment(input)
        assert len(output) == 1 and output[0] == input
        
    def test_on_none(self):
        pass

    def test_on_one_component(self):
        pass

    def test_on_several_components(self):
        pass

    def test_all_black_white_boarder(self):
        pass

    def test_text_letters_off_edge(self):
        pass
