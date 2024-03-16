from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np


def get_bboxes(img: Image) -> list[(int, int, int, int)]:
    result = []
    boxes = pytesseract.image_to_data(img, output_type=Output.DICT)
    for i in range(len(boxes['level'])):
        x1 = int(boxes['left'][i])
        y1 = int(boxes['top'][i])
        x2 = int(boxes['left'][i]) + int(boxes['width'][i])
        y2 = int(boxes['top'][i]) + int(boxes['height'][i])
        result.append((x1, y1, x2, y2))
    return result


def draw_bboxes(frame: Image, bboxes: list[(int, int, int, int)]):
    """Takes in a list of bounding boxes, and draws them on the given frame
    returns: Image"""
    frame = np.array(frame)
    for box in bboxes:
        frame = cv2.rectangle(frame, (box[0], box[1]),
                              (box[2], box[3]), (255, 0, 0))
    return frame


if __name__ == "__main__":
    test_img = Image.open('detection-dataset/20240128_090136.jpg')
    boxes = get_bboxes(test_img)
    annotated_img = draw_bboxes(test_img, boxes)
    annotated_img = cv2.resize(
        annotated_img, (int(annotated_img.shape[1]/4), int(annotated_img.shape[0]/4)))
    cv2.imshow("test", annotated_img)
    cv2.waitKey(0)
