import numpy as np
from model.basemodel import BaseModel

class BaseWordSegmenter():

    def __init__(self, model: BaseModel):
        self.model = model

    def segment(self, image: np.ndarray):
        raise NotImplementedError()