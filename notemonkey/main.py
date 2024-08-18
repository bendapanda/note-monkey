import numpy as np
import cv2

from line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
from word_segmenter.dpwordsegmenter import DPWordChunkSegmenter
from word_segmenter.connected_component_segmenter import ConnectedComponentWordSegmenter
from model.emnist_classifier import EMNISTModel
from model.basemodel import BaseModel
from chunk import Chunk
from line import Line
from imagehandler import ImageHandler, DeliveryMode
import preprocessor

if __name__ == "__main__":
    model = EMNISTModel('models/model_v5.keras', 'models/class_mapping.txt', verbosity=0)
    #model = BaseModel()
    word_segmenter = ConnectedComponentWordSegmenter(model, verbosity=0)
    line_segmenter = ProcessingLineSegmenter(word_segmenter, verbosity=0)
    handler = ImageHandler("./datasets/test")
    handler.image_delivery_mode = DeliveryMode.IN_ORDER
    for i in range(3):
        image = handler.get_new_image()
        image = handler.get_new_image()
        
        desired_width = 1500
        desired_height = int(image.shape[0] * (desired_width / image.shape[1]))
        image = cv2.resize(image, ( desired_width, desired_height))

        lines = line_segmenter.segment(preprocessor.resize_img(image, resize_factor=1))
        output = ""
        for line in lines:
            # currently I am assuming that the text can be perfectly segmented, but
            # this does not happen all the time currently
            for word in line.chunks:
                for chunk in word:
                    output+= chunk.value
                output += " "
            output += "\n"
        print(output)
        handler.show_image(preprocessor.resize_img(image, resize_factor=1))
    