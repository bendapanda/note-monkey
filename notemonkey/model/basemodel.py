import keras
import numpy as np
import cv2
class BaseModel():
    def __init__(self, verbosity=0):
        self.verbosity = verbosity

    def predict(self, image: np.ndarray):
        return None
    def _preprocess():  
        raise NotImplementedError

    def _postprocess():
        raise NotImplementedError