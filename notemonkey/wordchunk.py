"""class that contains the smallest unit of segmented code,
a connected piece of text."""
import numpy as np
import cv2
from model.basemodel import BaseModel

class Chunk():
    def __init__(self, image: np.ndarray, classifier: BaseModel):
        self.image = image
        self.value = classifier.predict(image)
