"""Handles line segmentation

Current to-implement list:
    - some way of handling the dot on i's
    - way to detetatch lines intersected by loopy lower letters
    - way to re-combine overly segmented lines

Author: Ben Shirley
Date: 2024-02-24
"""

import numpy as np
import cv2
from collections import deque
import preprocessor
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema


class Line():
    """class to represent an individual line
    of handwritten text"""

    def __init__(self, image: np.ndarray, chunks: set[tuple], chunk_width):
        self.original_image = image
        self.chunks = chunks
        self.chunk_width = chunk_width
        self.line_img = self._create_component_image(image, chunks)

    def _create_component_image(self, img, chunks):
        min_lane = min(chunks, key=lambda x: x[0])[0]
        max_lane = max(chunks, key=lambda x: x[0])[0]

        x1 = int(min_lane * self.chunk_width)
        x2 = int((max_lane + 1) * self.chunk_width)
        y1 = min(chunks, key=lambda x: x[1])[1]
        y2 = max(chunks, key=lambda x: x[2])[2]

        chunks_img = np.ones((y2-y1, x2-x1)) * 255
        for chunk in chunks:
            chunk_x1 = int(chunk[0] * self.chunk_width)
            chunk_x2 = int((chunk[0] + 1) * self.chunk_width)
            chunk_y1 = chunk[1]
            chunk_y2 = chunk[2]

            local_x1 = chunk_x1 - x1
            local_x2 = chunk_x2 - x1
            local_y1 = chunk_y1 - y1
            local_y2 = chunk_y2 - y1

            chunks_img[local_y1:local_y2,
                       local_x1:local_x2] = img[chunk_y1:chunk_y2, chunk_x1: chunk_x2]
        return chunks_img.astype(np.uint8)

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
        min_lane = min(self.chunks, key=lambda x: x[0])[0]
        max_lane = max(self.chunks, key=lambda x: x[0])[0]

        x1 = int(min_lane * self.chunk_width)
        x2 = int((max_lane + 1) * self.chunk_width)
        y1 = min(self.chunks, key=lambda x: x[1])[1]
        y2 = max(self.chunks, key=lambda x: x[2])[2]

        for chunk in self.chunks:
            chunk_x1 = int(chunk[0] * self.chunk_width)
            chunk_x2 = int((chunk[0] + 1) * self.chunk_width)
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
                      




class LineSegmenter():
    """class that handles the segmentation
    of an image into lines"""

    chunk_percentage = 0.05

    def segment(self, image: np.ndarray, verbosity=0) -> list[Line]:
        """segments an image and returns line objects"""
        proccessed_img = preprocessor.preprocess_img(image)

        blurred_img = preprocessor.blur_image(proccessed_img, (100, 20))
        columns, chunk_width = self._split_image(blurred_img)
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
            word_layout += [indexes]

        graph = self._construct_graph(word_layout)
        components = self._traverse_graph(graph)
        labels = self._create_lines(components, proccessed_img, chunk_width)
        return labels

    def _create_lines(self, components, image, chunk_width):
        """creates a list of word labels"""
        lines = []
        for component in components:
            lines.append(Line(image, component, chunk_width))
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

    def _split_image(self, image: np.ndarray) -> list[np.ndarray]:
        """splits an image into verticle columns of
        chunk_percentage"""
        width = image.shape[1]
        chunk_width = int(width * self.chunk_percentage)
        chunks = []
        index = 0
        while index+chunk_width < width:
            chunks.append(image[:, index:index+chunk_width])
            index += chunk_width
        chunks.append(image[:, index+chunk_width:])

        return chunks, chunk_width

    def _get_empty_lines(self, image: np.ndarray) -> np.ndarray:
        """returns an array of 1s and zeros depending on if
            the lines are empty"""
        result = np.zeros(image.shape[0])
        for i in range(image.shape[0]):
            result[i] = (np.all(image[i, :] >= 255*0.9))
        return result

    def _construct_graph(self, chunk_layout):
        """returns an adjacency dictionary for the chunks"""
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
    
    def clean_up_lines(self, lines):
        """oftentimes there are some issues with how lines are detected
        3 further processing steps are required:
        1. splitting connected layers
        2. combining chunks on the same line
        3. removing/recombining degenerate lines"""
        pass

        for line in lines:
            locations = self._get_number_lines(line)
            if len(locations) > 1:
                classified_chunks = self._assign_chunks_to_line(line, locations)

if __name__ == "__main__":
    image = cv2.imread("./misc-images/Selection_016.png")
    s=LineSegmenter()
    lines = s.segment(image)
    for line in lines:
        line.show(resize_factor=1)