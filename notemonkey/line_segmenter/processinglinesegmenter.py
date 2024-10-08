"""Handles line segmentation using a methodology described here:
need to find the paper hehe

overall I am not happy with the methods used here. I need to go back and consider other papers.

Author: Ben Shirley
Date: 2024-02-24
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from collections import deque
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
from copy import deepcopy

from line import Line
from word_segmenter.base_word_segmenter import BaseWordSegmenter
from line_segmenter.linesegmenter import LineSegmenter
import preprocessor
                     




class ProcessingLineSegmenter(LineSegmenter):
    """class that handles the segmentation
    of an image into lines"""

    def __init__(self, word_segmenter: BaseWordSegmenter, verbosity:int = 0 ):
        super().__init__(word_segmenter, verbosity=verbosity)
        self.chunk_percentage = 0.2
    

    def segment(self, image: np.ndarray) -> list[Line]:
        """segments an image and returns line objects
        
        Broadly follows the following steps:
        1. split the image into columns of chunk percentage
        2. assign each row of those columns either black or white
        3. process those columns to clean up inperfections
        4. construct a graph connecting black rectangles
        5. use that graph to find the connected lines
        """
        # preprocessing step
        proccessed_img = preprocessor.preprocess_img(image)

        blurred_img = preprocessor.blur_image(proccessed_img, (100, 20))
        if self.verbosity >= 3:
            cv2.imshow("preprocessed image",preprocessor.resize_img(blurred_img))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # split the image into columns
        columns, chunk_width = self._split_image(blurred_img)
        # decide whether each row of each column should be black or white
        painted_columns = []
        for column in columns:
            painted_col = self._paint_column(column)
            painted_col = painted_col.astype(np.uint8)
            painted_col = preprocessor.otsu_thresholding(painted_col)
            painted_columns.append(painted_col)
        
            
        if self.verbosity >= 3:
            cv2.imshow("binarized image", preprocessor.resize_img(np.hstack(painted_columns)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        smoothed_columns = self._remove_small_white_boxes(painted_columns) 
        smoothed_columns = self._remove_isolated_black_boxes(smoothed_columns)
        if self.verbosity >= 3:
            cv2.imshow("smoothed", preprocessor.resize_img(np.hstack(smoothed_columns)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # now we need to try to extend some of the lines that we chopped off when deleting the large black rectangles

        #columns = self._dialation_operation(columns)

        # We need to process these chunks now
        word_layout = []
        for column in columns:
            indexes = []
            empty = self._get_empty_lines(column)
            x_start = 0
            index = 0
            started = False
            for pixel in empty:
                if not started and pixel == 0:
                    x_start = index
                    started = True
                elif started and pixel == 1:
                    indexes.append((x_start, index))
                    started = False
                index += 1
            if started:
                indexes.append((x_start, len(empty)))
            word_layout += [indexes]

        # now that we have processed the chunks we can get all connected components
        graph = self._construct_graph(word_layout)
        components = self._traverse_graph(graph)
        lines = self._create_lines(components, proccessed_img, chunk_width)

        return lines
    
    def _paint_column(self, column: np.ndarray) -> np.ndarray:
        """Takes a column of the image as input,
        sets each row of a matching column to be the average of the input column"""
        painted_column = np.ones(column.shape)
        for i in range(column.shape[0]):
            column_color = np.sum(column[i, :]) / len(column[i, :])
            painted_column[i, :] *= column_color
        return painted_column
    
         
    
    def _get_boxes(self, column: np.ndarray, colour:int):
        """returns start and stop coordinates of all the boxes of that colour
        in the column image"""
        if colour not in [255, 0]:
            raise ValueError("colour can only be black or white")

        column_boxes = []
        started = None
        for i in range(column.shape[0]):
            if column[i][0] == colour and started is None:
                started = i
            elif column[i][0] != colour and started is not None:
                column_boxes.append((started, i-1))
                started = None
        if started is not None:
            column_boxes.append((started, column.shape[0]))
        
        return column_boxes
    
    def _remove_small_white_boxes(self, columns:list[np.ndarray]):
        """Goes through the columns of our processed image and fills in white rectangles
        that are unusually small
        
        This is done by fillling in all boxes smaller than a factor of the median"""
        all_white_boxes = []
        all_white_heights = []

        for column in columns:  
            white_boxes = self._get_boxes(column, 255)

            white_heights = [box[1] - box[0] for box in white_boxes]
            white_heights = np.array(white_heights)

            all_white_heights.append(white_heights)
            all_white_boxes.append(white_boxes)


        #removing small while sections
        if self.verbosity>=4:
            # display line statistics
            data = np.concatenate(all_white_heights)
            kde = KernelDensity(bandwidth=50).fit(data.reshape(-1, 1))
            x = np.linspace(0, np.max(data), 300)
            y = kde.score_samples(x.reshape(-1, 1))

            mode_index = np.argmax(y)

            plt.plot(x, np.exp(y)*2000, label="kde graph")
            plt.plot(x[mode_index], np.exp(y[mode_index])*2000, 'x')
            plt.plot(np.median(data), 0, 'o')
            # IDEA: Maybe it's a good idea to try second peak as our metric

            plt.hist(data)
            plt.show()
            
        global_median = np.median(np.concatenate(all_white_heights))
        MEADIAN_SCALING_FACTOR = 0.5
        for i in range(len(columns)):
            for box in all_white_boxes[i]:
                if box[1]-box[0] < global_median*MEADIAN_SCALING_FACTOR:
                    # fill box black
                    columns[i][box[0]:box[1]+1, :] = 0
        
        if self.verbosity>=4:
            cv2.imshow("white chunks removed", preprocessor.resize_img(np.hstack(columns)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return columns

    def _remove_isolated_black_boxes(self, columns: list[np.ndarray]) -> list[np.ndarray]:
        """Second cleanup step. We need to remove the black boxes that are isolated from other boxes."""
        all_black_boxes = []
        all_black_heights = []

        for column in columns:  
            black_boxes = self._get_boxes(column, 0)

            black_heights = [box[1] - box[0] for box in black_boxes]
            black_heights = np.array(black_heights)

            all_black_boxes.append(black_boxes)
            all_black_heights.append(black_heights)

        # removing dangling black boxes
        all_black_heights = np.concatenate(all_black_heights)
        all_black_heights = np.sort(all_black_heights)
        NUMBER_SEGMENTS = 8 # as chosen in the paper
        if len(all_black_heights) >= NUMBER_SEGMENTS:
            parts = np.array_split(all_black_heights, NUMBER_SEGMENTS)
            T = np.sum([np.max(part) for part in parts]) / NUMBER_SEGMENTS
        else:
            T = np.sum(all_black_heights) / len(all_black_heights)
        
        for i in range(len(columns)):
            for box in all_black_boxes[i]:
                if box[1]-box[0] < 0.5*T:
                    columns[i][box[0]:box[1]+1, :] = 255

        if self.verbosity>=4:
            cv2.imshow("dangling black boxes removed", preprocessor.resize_img(np.hstack(columns)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()


        return columns

    def _create_lines(self, components, image, chunk_width):
        """creates a list of word labels"""
        lines = []
        for component in components:
            line_img = self._create_line_image(image, component, chunk_width)
            lines.append(Line(line_img, self.word_segmenter))
        return lines

    def _traverse_graph(self, graph):
        """traverses the component graph and returns sets
        of connected components"""
        components = []
        state = {node: "U" for node in graph.keys()}
        for node in state.keys():
            if state[node] == "U":
                queue = deque()
                queue.appendleft(node)
                component, state = self._bfs_loop(queue, graph, state, {node})
                components.append(component)
        return components

    def _bfs_loop(self, queue, graph, state, component):
        """The bfs part of the graph traversal"""
        while len(queue) != 0:
            node = queue.pop()
            for v in graph[node]:
                if state[v] == "U":
                    state[v] = "D"
                    component.add(v)
                    queue.appendleft(v)
            state[node] = "P"
        return component, state

    def _dialation_operation(self, columns: np.ndarray, connection_distance=2):
        """when we dialate boxes, we need to do two things:
        First, make sure that horizontal cavities are filled,
        so holes in chunks, or divots in the horizontal direction.
        
        Then, we need to connect parts that should be connected. To do this, identify
        left/right edges, and test to see if there are any intersections within
        say 2 columns of that point"""
        # remove large black chunks
        columns = self._remove_large_black_boxes(columns)
        columns = self._fill_horizontal_cavities(columns)
        columns = self._connect_disconnected_chunks(columns, connection_distance)
        columns = self._extend_edge_chunks(columns, extension_distance=connection_distance)

        return columns
    
    def _remove_large_black_boxes(self, columns: list[np.ndarray]) -> list[np.ndarray]:
        """For use in the dialation operation. 
        Large black boxes can result in two lines merging, so we need to remove them."""
        columns = deepcopy(columns)
        all_black_heights = []
        all_black_boxes = []
        for column in columns:
            black_boxes = self._get_boxes(column, 0)
            black_heights = [box[1]-box[0] for box in black_boxes]
            all_black_boxes.append(black_boxes)
            all_black_heights += black_heights
        all_black_heights = np.array(all_black_heights)

        avg_height = np.mean(all_black_heights)
        for i in range(len(columns)):
            for box in all_black_boxes[i]: 
                if box[1]-box[0] >= 2*avg_height:
                    columns[i][box[0]:box[1]+1, :] = 255
        return columns

    def _extend_edge_chunks(self, columns, extension_distance=1):
        """It is beneficial to extend lines that finish before the
        end of the image. So, we locate boxes that are isolated on the left and right,
        and then so long as they are not connected by the extension, we make it."""
        boxes = []
        for i in range(len(columns)):
            boxes.append(self._get_boxes(columns[i], 0))
        
        #TODO allow multiple extensions, not just one
        for column_index in range(len(boxes)):
            for box_index in range(len(boxes[column_index])):

                isolated_left = True
                if column_index > 0:
                    for other_box in boxes[column_index-1]:
                        if self._touching((column_index-1,)+other_box, (column_index,)+boxes[column_index][box_index]):
                            isolated_left = False
                else:
                    isolated_left = False
                
                isolated_right = True
                if column_index < len(boxes) - 1:
                    for other_box in boxes[column_index+1]:
                        if self._touching((column_index+1,)+other_box, (column_index,)+boxes[column_index][box_index]):
                            isolated_right  = False
                else:
                    isolated_right = False

                if isolated_left:
                    point_to_extend = int((boxes[column_index][box_index][1] + boxes[column_index][box_index][0]) / 2)
                    columns[column_index-1][point_to_extend-1:point_to_extend+1, :] = 0
                if isolated_right:
                    point_to_extend = int((boxes[column_index][box_index][1] + boxes[column_index][box_index][0]) / 2)
                    columns[column_index+1][point_to_extend-1:point_to_extend+1, :] = 0
        return columns



    def _connect_disconnected_chunks(self, columns, connection_distance):
        # Now, connect disconnected chunks
        #loop over all columns and boxes
        # if box is disconnected on one side, look at the next one.
        # if it could be connected there, connect them
        new_columns = deepcopy(columns)
        boxes = []
        for i in range(len(new_columns)):
            boxes.append(self._get_boxes(new_columns[i], 0))
        for column_index in range(len(new_columns)):
            for box_index in range(len(boxes[column_index])):
                left_rows_to_check = []
                right_rows_to_check = []
                for offset in range(1, connection_distance+1):
                    if column_index - offset >= 0:
                        left_rows_to_check.append(boxes[column_index - offset])
                    if column_index + offset <= len(boxes) - 1:
                        right_rows_to_check.append(boxes[column_index + offset])

                left_index = 0
                connection_made = False
                left_box_to_connect = None
                while not connection_made and left_index < len(left_rows_to_check):
                    for other_box in left_rows_to_check[left_index]:
                        if self._touching((left_index,)+other_box, (box_index,) + boxes[column_index][box_index]):
                            connection_made = True
                            left_box_to_connect = (column_index-left_index-1, other_box[0], other_box[1])
                    left_index += 1
                
                right_index = 0
                connection_made = False
                right_box_to_connect = None
                while not connection_made and right_index < len(right_rows_to_check):
                    for other_box in right_rows_to_check[right_index]:
                        if self._touching((right_index,)+other_box, (box_index,)+boxes[column_index][box_index]):
                            
                            connection_made = True
                            right_box_to_connect = (column_index+right_index+1, other_box[0], other_box[1])
                    right_index += 1

                # Now, connect left and right boxes to connect if they are not already connected
                if left_box_to_connect is not None and abs(column_index - left_box_to_connect[0]) > 1:
                    start_point_to_draw_through = int(max(left_box_to_connect[1], boxes[column_index][box_index][0]))
                    end_point_to_draw_through = int(min(left_box_to_connect[2], boxes[column_index][box_index][1]))
                    for offset in range(1, abs(column_index - left_box_to_connect[0])):
                        new_columns[column_index - offset][start_point_to_draw_through:end_point_to_draw_through, :] = 0

                if right_box_to_connect is not None and abs(column_index - right_box_to_connect[0]) > 1:
                    start_point_to_draw_through = int((max(right_box_to_connect[1], boxes[column_index][box_index][0])))
                    end_point_to_draw_through = int(min(right_box_to_connect[2], boxes[column_index][box_index][1]))
                    for offset in range(1, abs(column_index + right_box_to_connect[0])):
                        new_columns[column_index - offset][start_point_to_draw_through:end_point_to_draw_through, :] = 0
        
        return new_columns


    def _fill_horizontal_cavities(self, columns:np.ndarray):
        """takes in a list of columns, and fills in any horizontal overhangs"""
        #First, fill cavities:
        #basically we are looking for 2 vertically stacked boxes that are both touching the same adjacent box
        new_columns = deepcopy(columns)
        old_columns = [np.zeros_like(column) for column in new_columns]
        while any(np.any(new_columns[i] != old_columns[i]) for i in range(len(new_columns))):
            old_columns = new_columns
            new_columns = deepcopy(new_columns)
            # go through each column and find pairs
            boxes = []
            for i in range(len(new_columns)):
                boxes.append(self._get_boxes(new_columns[i], 0))

            for column_index in range(len(new_columns)):          
                for box_index in range(len(boxes[column_index]) - 1):
                    top_box = boxes[column_index][box_index]
                    bottom_box = boxes[column_index][box_index+1]

                    boxes_to_consider = []
                    if column_index > 0:
                        boxes_to_consider += boxes[column_index - 1]
                    if column_index < len(new_columns) - 1:
                        boxes_to_consider += boxes[column_index+1]

                    boxes_connected = False
                    for other_box in boxes_to_consider:
                        if self._touching((i-1,)+ other_box, (i,)+top_box) and self._touching((i-1,) +other_box, (i,)+bottom_box):
                            boxes_connected = True
                    
                    if boxes_connected:
                        new_columns[column_index][top_box[1]:bottom_box[0], :] = 0
        return new_columns


    def _split_image(self, image: np.ndarray) -> list[np.ndarray]:
        """splits an image into verticle columns of
        chunk_percentage"""
        width = image.shape[1]
        chunk_width = max(1, int(width * self.chunk_percentage))
        chunks = []
        index = 0
        while index+chunk_width < width:
            chunks.append(image[:, index:index+chunk_width])
            index += chunk_width
        chunks.append(image[:, index:])

        return chunks, chunk_width

    def _get_empty_lines(self, image: np.ndarray) -> np.ndarray:
        """go through each row of the column,
        returns an array with ones if the row is all white, and
        0 otherwise"""
        result = np.zeros(image.shape[0])
        for i in range(image.shape[0]):
            result[i] = (np.all(image[i, :] >= 255*0.9))
        return result

    def _construct_graph(self, chunk_layout):
        """returns an adjacency dictionary for the chunks
        The point of this method is that we want to get the connected components"""
        graph = {}
        for lane in range(len(chunk_layout)):
            for y1, y2 in chunk_layout[lane]:
                touching = self._get_touching((lane, y1, y2), chunk_layout)
                for chunk in touching:
                    graph[chunk] = graph.get(chunk, []) + [(lane, y1, y2)]
                graph[(lane, y1, y2)] = graph.get(
                    (lane, y1, y2), []) + touching
        return graph

    def _get_touching(self, box, chunk_layout):
        """returns a list of all boxes the given box is touching"""

        next_lane = box[0] + 1
        if next_lane >= len(chunk_layout):
            return []
        results = []
        for other_box in chunk_layout[box[0]+1]:
            if self._touching(box, (next_lane, other_box[0], other_box[1])):
                results.append((next_lane, other_box[0], other_box[1]))
        return results

    def _touching(self, box1, box2):
        """takes in (lane, y1, y2) tuples, returns true if they are touching"""
        return abs(box1[0]-box2[0]) == 1 and\
            ((box1[1] <= box2[1] and box1[2] >= box2[1]) or
             (box1[2] <= box2[2] and box1[2] >= box2[1]) or
             (box2[1] <= box1[2] and box2[1] >= box1[1]) or
             (box2[2] <= box1[2] and box2[2] >= box1[1]))
    
    def _get_chunk_centers(self, line: Line) -> list[tuple]:
        """returns a list of center tuples for each chunk"""
        result = []
        for chunk in line.chunks:
            result.append((int((chunk[0]+0.5)*line.chunk_width),
                           (int(chunk[1]+chunk[2])/2)))
        return result
    def _get_number_lines(self, line, smoothing=0.1) -> int:
        """sometimes we have connecting bounding boxes when we shouldn't,
        due to loopy letters. we can analyse the height of each bbox to 
        find out when this has occured."""
        y_values = np.array([x[1] for x in self._get_chunk_centers(line)]).reshape(-1, 1)
        normalized = y_values / np.max(y_values)
        data = np.sort(normalized)
        kde = KernelDensity(bandwidth=smoothing).fit(data.reshape(-1, 1))
        x  = np.linspace(0, 1, 100)
        y = kde.score_samples(x.reshape(-1, 1))

        peaks = argrelextrema(y, np.greater)[0]
        return x[peaks]
    
    def _assign_chunk_to_line(self, line, locations):
        """given a list of locations, goes through each chunk and
        partitions the set to appropriately match the list of line locations"""
        locations *= np.max(self._get_chunk_centers(line), key=lambda x:x[1])

        for chunk in line.chunks:
            pass
    
    def _create_line_image(self, img, pieces, piece_width):
        min_lane = min(pieces, key=lambda x: x[0])[0]
        max_lane = max(pieces, key=lambda x: x[0])[0]

        x1 = int(min_lane * piece_width)
        x2 = int((max_lane + 1) * piece_width)
        y1 = min(pieces, key=lambda x: x[1])[1]
        y2 = max(pieces, key=lambda x: x[2])[2]

        pieces_img = np.ones((y2-y1, x2-x1)) * 255
        for piece in pieces:
            chunk_x1 = int(piece[0] * piece_width)
            chunk_x2 = int((piece[0] + 1) * piece_width)
            chunk_y1 = piece[1]
            chunk_y2 = piece[2]

            local_x1 = chunk_x1 - x1
            local_x2 = chunk_x2 - x1
            local_y1 = chunk_y1 - y1
            local_y2 = chunk_y2 - y1

            pieces_img[local_y1:local_y2,
                       local_x1:local_x2] = img[chunk_y1:chunk_y2, chunk_x1: chunk_x2]
        return pieces_img.astype(np.uint8)

 