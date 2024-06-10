import numpy as np
import cv2

from line_segmenter.processinglinesegmenter import ProcessingLineSegmenter
from word_segmenter.dpwordsegmenter import DPWordChunkSegmenter
from line import Line
from imagehandler import ImageHandler
import preprocessor

if __name__ == "__main__":
    line_segmenter = ProcessingLineSegmenter(DPWordChunkSegmenter())
    handler = ImageHandler("detection-dataset")
    for i in range(20):
        image = handler.get_new_image()
        handler.show_image(image)
        lines = line_segmenter.segment(image)
        for line in lines:
            handler.show_image(preprocessor.resize_img(line.line_img))
            for chunk in line.chunks:
                handler.show_image(preprocessor.resize_img(chunk))