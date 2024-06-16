import numpy as np
from context import notemonkey
import pytest

from notemonkey.word_segmenter.dpwordsegmenter import DPWordChunkSegmenter
from  notemonkey.model.basemodel import BaseModel

class TestDPWordChunkSegmenter():
    segmenter = DPWordChunkSegmenter(BaseModel())
    
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
        assert len(output) == 1 and output[0].image.shape == input.shape and np.all(output[0].image == input)
        
    # these are all operational tests, not on tests with specific bugs that could arise
    # so leaving as stubs for now.
    def test_on_one_component(self):
        pass

    def test_on_several_components(self):
        pass

    def test_all_black_white_boarder(self):
        pass

    def test_text_letters_off_edge(self):
        pass
