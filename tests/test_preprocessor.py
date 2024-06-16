import numpy as np 
from context import notemonkey

import pytest
import notemonkey.preprocessor as preprocessor

class TestPreprocessor():
    def test_crop_tight_none(self):
        try: 
            preprocessor.crop_image_tight(None)
            pytest.fail()
        except ValueError:
            pass
    
    def test_crop_tight_trivial_image(self):
        input = np.array([[]])
        output = preprocessor.crop_image_tight(input)
        assert output.shape == (1,0)
        
    def test_crop_tight_all_white(self):
        input = np.ones((100, 100), dtype=np.float32)
        output = preprocessor.crop_image_tight(input)
        assert output.shape == (1, 0)

    def test_crop_tight_255_value(self):
        input = np.ones((100,100), np.uint8)*255
        output = preprocessor.crop_image_tight(input)
        assert output.shape == (1, 0)

    def test_crop_tight_all_black(self):
        input = np.zeros((100,100), np.uint8)
        output = preprocessor.crop_image_tight(input)
        assert  np.all(output == input)

    def test_resize_img(self): 
        input = np.ones((100,100), np.float32)
        output = preprocessor.resize_img(input)
        assert output.shape == (25, 25)
    
    def test_resize_img_scale_factor(self):
        input = np.ones((100,100), np.float32)
        output = preprocessor.resize_img(input, resize_factor=1)
        assert output.shape == (100,100)

    def test_resize_one_pixel(self):
        input = np.array([[255]])
        try: 
            output = preprocessor.resize_img(input)
            pytest.fail()
        except ValueError:
            pass 
        
    def test_dialate_by_density_3channel(self):   
        input = np.ones((100,100,3))*255
        try: 
            output = preprocessor.dialate_by_pixel_density(input)
            pytest.fail()
        except ValueError:
            pass
    def test_dialate_by_density_non_binarized(self):   
        input = np.ones((100,100))*255
        input[0,0] = 1
        input[1,0] = 3
        try: 
            output = preprocessor.dialate_by_pixel_density(input)
            pytest.fail()
        except ValueError:
            pass   

    def test_dialate_pixel_density_trivial(self):
        input = np.array([[255]]) 
        try:
            output = preprocessor.dialate_by_pixel_density(input)
            pytest.fail()
        except ValueError:
            pass

    def test_preprocess_1pixel(self):
        img = np.array([[[255,255,255]]], np.uint8)     
        output = preprocessor.preprocess_img(img)
        assert np.all(output == img)
        
    def test_greyscale_image(self):
        img = np.array([[255,0,255]], np.uint8)
        output = preprocessor.preprocess_img(img)
        assert np.all(img == output)

    def test_empty_image(self):
        img = np.array([[]])
        try:
            output = preprocessor.preprocess_img(img)
            pytest.fail()
        except ValueError:
            pass

    def test_oned_array(self):
        img = np.array([1,2,3,4,5])
        try:
            output = preprocessor.preprocess_img(img)
            pytest.fail()
        except ValueError:
            pass
