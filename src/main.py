import numpy as np
import cv2

from line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
from word_segmenter.dpwordsegmenter import DPWordChunkSegmenter
from model.emnist_classifier import EMNISTModel
from chunk import Chunk
from line import Line
from imagehandler import ImageHandler, DeliveryMode
import preprocessor

if __name__ == "__main__":
    line_segmenter = ProcessingLineSegmenter(DPWordChunkSegmenter(EMNISTModel('./models/model_v5.keras', './models/class_mapping.txt')))
    handler = ImageHandler("./datasets/test")
    handler.image_delivery_mode = DeliveryMode.IN_ORDER
    for i in range(10):
        image = handler.get_new_image()
        image = handler.get_new_image()
        #handler.show_image(preprocessor.resize_img(image))
        image = preprocessor.resize_img(image, resize_factor=32)
        lines = line_segmenter.segment(image)
        output = ""
        for line in lines:
            #handler.show_image(preprocessor.resize_img(line.line_img))
            for chunk in line.chunks:
                #handler.show_image(preprocessor.resize_img(chunk.image))
                output+= " " +  chunk.value
            output += "\n"
        print(output)
        handler.show_image(preprocessor.resize_img(image))