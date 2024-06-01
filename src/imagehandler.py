"""
Class that handles the loading and reading of image files, for the purposes of
running tests and displaying image files
"""

from enum import Enum
import os
import cv2
import random
import numpy as np

class DeliveryMode(Enum):
    IN_ORDER = 0
    RANDOM = 1
    REVERSE_ORDER = 2


class ImageHandler():

    def __init__(self, base_directory: str) -> None:
        self.filepath = base_directory
        self.current_index = 0
        self.image_delivery_mode = DeliveryMode.RANDOM
    
    def get_new_image(self):
        image_names = os.listdir(self.filepath) 
        if self.image_delivery_mode == DeliveryMode.IN_ORDER:
           image = cv2.imread(f"{self.filepath}/{image_names[self.current_index]}")
           self.current_index += 1
           self.current_index %= len(image_names)
           return image
        elif self.image_delivery_mode == DeliveryMode.RANDOM:
            retreval_index = random.randint(0, len(image_names)-1)
            image = cv2.imread(f"{self.filepath}/{image_names[retreval_index]}")
            return image
    
    def show_image(self, image: np.ndarray):
        cv2.imshow("image", image)
        cv2.waitKey(0)

if __name__ == "__main__":
    handler = ImageHandler("detection-dataset")
    for i in range(6):
        image = handler.get_new_image()
        handler.show_image(image)