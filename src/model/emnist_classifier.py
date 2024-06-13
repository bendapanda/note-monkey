import numpy as np
import cv2
from model.basemodel import BaseModel
import preprocessor

class EMNISTModel(BaseModel):
    def __init__(self, filepath: str, class_mapping_path: str):
        super().__init__( filepath)
        self.class_mapping = {}
        with open(class_mapping_path) as file:
            for line in file:
                index, character_code = line.split()
                self.class_mapping[int(index)] = chr(int(character_code))

    def _preprocess(self, image):
        #need to convert the image to 24x24 white on black
        # assume it is already greyscale and binarised
        #TODO have checks for this 
        # before we resize we need to crop tight, add some padding, and make it square
        cropped = preprocessor.crop_image_tight(image)
        padding_size = int(cropped.shape[0] * 0.1)
        padded_image = (np.ones((cropped.shape[0] + 2*padding_size, cropped.shape[1]))*255).astype(int)
        padded_image[padding_size:padding_size+cropped.shape[0], :] = cropped
        # Now, to add side padding to make it square

        desired_width = max(padded_image.shape[0], padded_image.shape[1])
        squared_image = (np.ones((padded_image.shape[0], desired_width))*255).astype(int)
        #find where we should insert the old image
        x_start = int(max(squared_image.shape[1]/2 - padded_image.shape[1]/2, 0))
        squared_image[:, x_start:x_start+padded_image.shape[1]] = padded_image

        resized = cv2.resize(squared_image.astype(float), (28, 28))
        normalised = resized / np.max(resized)
        inverted = 1 - normalised
        inverted = np.fliplr(inverted)
        rotated = cv2.rotate(inverted, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # I want to try dialate the image to make the text a bit thicker
        # Define a kernel for dilation
        kernel = np.ones((3, 3), np.uint8) 
        # Dilate the image
        dilated_image = cv2.dilate(rotated, kernel, iterations=1)
        softened_image = cv2.GaussianBlur(dilated_image, (3,3), 0)

        return softened_image 
    
    def _postprocess(self,  model_output):
        predicted_class = np.argmax(model_output)
    
        return self.class_mapping[predicted_class]