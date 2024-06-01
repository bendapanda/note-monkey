from src.line import Line, LineSegmenter
import src.preprocessor as preprocessor
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

filepath = "./detection-dataset/20240128_090136.jpg"


def show_lines(filepath):
    img = cv2.imread(filepath)

    segmenter = LineSegmenter()
    lines = segmenter.segment(img)
    for line in lines:
        line.show_thinning()
        

def test_line_numbers(filepath):
    img = cv2.imread(filepath)

    segmenter = LineSegmenter()
    lines = segmenter.segment(img)
    for line in lines:
        line.show()
        data = line.get_number_lines()
    

def test_blur(filepath):
    img = cv2.imread(filepath)
    img = preprocessor.preprocess_img(img)
    img = preprocessor.blur_image(img, (100, 20))
    img = preprocessor.resize_img(img)
    for line in img:
        print(np.all(line == 255))
    cv2.imshow("test", img)
    cv2.waitKey(0)

def test_thinning(filepath):
    img = cv2.imread(filepath)
    img = preprocessor.preprocess_img(img)
    img = preprocessor.thin_text(img)
    #img = preprocessor.resize_img(img)

    cv2.imshow("test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filenames = os.listdir("./detection-dataset/")
    for filename in filenames:
        show_lines("./detection-dataset/" + filename)
