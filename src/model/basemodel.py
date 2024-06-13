import keras
import numpy as np
import cv2
class BaseModel():
    def __init__(self, filepath:str):
        self.model =  keras.models.load_model(filepath)

    def predict(self, image: np.ndarray):
        processed_image = self._preprocess(image)
        prediction = self.model.predict(np.array([processed_image]))
        classification = self._postprocess(prediction)
        
        verbosity= 2
        if verbosity >= 4:  
            cv2.imshow(f"class: {classification}",cv2.resize(processed_image, (100,100)))
            cv2.waitKey(0)
        return classification

    def _preprocess():  
        raise NotImplementedError

    def _postprocess():
        raise NotImplementedError