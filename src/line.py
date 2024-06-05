import preprocessor

import numpy as np
import cv2
from wordsegmenter import WordChunkSegmenter


class Line():
    """class to represent an individual line
    of handwritten text
    
    NOTE: I am not sure if line should be doing all this processing, feels like that should be handled by the segmenter"""

    def __init__(self, image: np.ndarray, pieces: set[tuple], piece_width, segmenter: WordChunkSegmenter):
        self.original_image = image
        self.pieces = pieces
        self.piece_width = piece_width
        self.line_img = self._create_component_image(image)
        
        self.segmenter = segmenter
        self.chunks = segmenter.segment(self.line_img)

    def _create_component_image(self, img):
        min_lane = min(self.pieces, key=lambda x: x[0])[0]
        max_lane = max(self.pieces, key=lambda x: x[0])[0]

        x1 = int(min_lane * self.piece_width)
        x2 = int((max_lane + 1) * self.piece_width)
        y1 = min(self.pieces, key=lambda x: x[1])[1]
        y2 = max(self.pieces, key=lambda x: x[2])[2]

        pieces_img = np.ones((y2-y1, x2-x1)) * 255
        for piece in self.pieces:
            chunk_x1 = int(piece[0] * self.piece_width)
            chunk_x2 = int((piece[0] + 1) * self.piece_width)
            chunk_y1 = piece[1]
            chunk_y2 = piece[2]

            local_x1 = chunk_x1 - x1
            local_x2 = chunk_x2 - x1
            local_y1 = chunk_y1 - y1
            local_y2 = chunk_y2 - y1

            pieces_img[local_y1:local_y2,
                       local_x1:local_x2] = img[chunk_y1:chunk_y2, chunk_x1: chunk_x2]
        return pieces_img.astype(np.uint8)

    def avg_layers(self) -> float:
        """takes in a line object and returns the average number of layers,
        that is, the average number of chunks stacked on top of each other
        per chunk"""
        rows = {}
        for chunk in self.chunks:
            rows[chunk[0]] = rows.get(chunk[0], 0) + 1
        return sum(rows.values())/len(rows.values())
    
    def draw_boxes(self, image) -> np.ndarray:
        """draws boxes on the imgae"""
        image_to_show = image.copy()
        image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2RGB)
        min_lane = min(self.pieces, key=lambda x: x[0])[0]
        max_lane = max(self.pieces, key=lambda x: x[0])[0]

        x1 = int(min_lane * self.chunk_width)
        x2 = int((max_lane + 1) * self.chunk_width)
        y1 = min(self.pieces, key=lambda x: x[1])[1]
        y2 = max(self.pieces, key=lambda x: x[2])[2]

        for chunk in self.pieces:
            chunk_x1 = int(chunk[0] * self.piece_width)
            chunk_x2 = int((chunk[0] + 1) * self.piece_width)
            chunk_y1 = chunk[1]
            chunk_y2 = chunk[2]

            local_x1 = chunk_x1 - x1
            local_x2 = chunk_x2 - x1
            local_y1 = chunk_y1 - y1
            local_y2 = chunk_y2 - y1

            image_to_show = cv2.rectangle(image_to_show,
                                          (local_x1, local_y1), (local_x2, local_y2),
                                          (255, 0, 0),
                                          4)
        return image_to_show
    
    def show(self, resize_factor=0.25):
        """displays line, with cluster locations on top"""

        image_to_show = self.draw_boxes(self.line_img)
        
        image_to_show = self.color_largest_rectangle(image_to_show)
        image_to_show = preprocessor.resize_img(
            image_to_show, resize_factor=resize_factor)
        cv2.imshow("line", image_to_show)
        cv2.waitKey(0)

    def show_thinning(self):
        image = preprocessor.thin_text(self.line_img)
        image = self.draw_boxes(image)

        cv2.imshow("text", image)
        cv2.waitKey(0)

    def show_on_original(self, resize_factor=0.25):
        """shows the bboxes for the chunk on the original_image"""
        image_to_show = self.original_image.copy()
        image_to_show = cv2.cvtColor(image_to_show, cv2.COLOR_GRAY2BGR)
        for chunk in self.chunks:
            image_to_show = cv2.rectangle(image_to_show,
                                          (int(
                                              chunk[0] * self.chunk_width), chunk[1]),
                                          (int(
                                              (chunk[0]+1) * self.chunk_width), chunk[2]),
                                          (255, 0, 0),
                                          4)
        image_to_show = preprocessor.resize_img(
            image_to_show, resize_factor=resize_factor)
        cv2.imshow("line", image_to_show)
        cv2.waitKey(0)

    def color_largest_rectangle(self, image):
        min_lane = min(self.chunks, key=lambda x: x[0])[0]
        max_lane = max(self.chunks, key=lambda x: x[0])[0]

        x1 = int(min_lane * self.chunk_width)
        x2 = int((max_lane + 1) * self.chunk_width)
        y1 = min(self.chunks, key=lambda x: x[1])[1]
        y2 = max(self.chunks, key=lambda x: x[2])[2]
        
        max_rectangle = None
        for chunk in self.chunks:
            if max_rectangle is None or max_rectangle[2]-max_rectangle[1]< chunk[2]-chunk[1]:
                max_rectangle = chunk

        chunk_x1 = int(max_rectangle[0] * self.chunk_width)
        chunk_x2 = int((max_rectangle[0] + 1) * self.chunk_width)
        chunk_y1 = max_rectangle[1]
        chunk_y2 = max_rectangle[2]

        local_x1 = chunk_x1 - x1
        local_x2 = chunk_x2 - x1
        local_y1 = chunk_y1 - y1
        local_y2 = chunk_y2 - y1
        
        new_img = cv2.rectangle(image, (local_x1, local_y1),
                            (local_x2, local_y2),
                            (0, 0, 255), 6)
        return new_img
 