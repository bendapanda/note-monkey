import numpy as np 
from context import notemonkey

import pytest
from notemonkey.line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
from notemonkey.word_segmenter.trivial_word_segmenter import TrivialWordSegmenter
from notemonkey.model.basemodel import BaseModel

class TestProcessingLineSegmenter():
    segmenter = ProcessingLineSegmenter(TrivialWordSegmenter(BaseModel()))

    def test_empty_image(self):
        img = np.ones((100,100, 3), np.uint8)*255
        output = self.segmenter.segment(img)
        assert len(output) == 0
    
    def test_all_black(self):
        img = np.zeros((100,100, 3),np.uint8)
        output = self.segmenter.segment(img)
        assert len(output) == 1
    
    def test_single_line(self):
        img = np.ones((100,100,3), np.uint8)*255
        img[25:45,10:90] = 0
        output = self.segmenter.segment(img)
        assert len(output) == 1

    def test_trivial_image(self):
        img = np.array([[[255, 255,255]]], np.uint8)
        print(img.shape)
        output = self.segmenter.segment(img)
        assert len(output) == 0

    def test_empty_image(self):
        img = np.array([[[]]], np.uint8)
        try:
            output = self.segmenter.segment(img)
            pytest.fail()
        except ValueError: 
            pass
    
    def test_black_white_image(self):
        pass