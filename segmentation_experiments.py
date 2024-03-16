"""
Experiments on algorithmic character segmentation

Author: Ben Shirley
Date: 21/2/2024
"""
import cv2
import numpy as np
from collections import deque

# preprocessing
# ==============================================


def resize_img(img, resize_factor=0.25):
    y, x = img.shape[:2]
    img = cv2.resize(img, (int(x*resize_factor), int(y*resize_factor)))
    return img


def preprocess_img(img):
    """
    preprocessing with the following steps:
        converting to grayscale
        binarizing
        thinning?
    """
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, binarized_img = cv2.threshold(
        grayscale_img, 128, 255, cv2.THRESH_BINARY)
    return binarized_img

# Line segmentation
# ===================================================
# idea 1: find the number of black pixels in each license
# idea 2: split the image into 5% chuncks, and repeat step 1 on each chuncks
# idea 3: idea 2 worked pretty well, so let's try connect all connected white
# chunks and then take the smallest box that includes all of them!
# TODO: find a way to encorperate dots, like in the 'i' into the components


class Cluster():
    """class to represent a cluster object"""
    lanes = {}
    max_lane = None
    min_lane = None
    min_y = None
    max_y = None

    def __init__(self, chunk, lane):
        self.lanes[lane] = set([chunk])
        self.max_lane = lane + 1
        self.min_lane = lane
        self.min_y = chunk[0]
        self.max_y = chunk[1]

    def add_to_lane(self, lane, chunk):
        self.lanes = self.lanes.get(lane, set()).union({chunk})
        self.min_y = min(self.min_y, chunk[0])
        self.max_y = max(self.max_y, chunk[1])
        self.min_lane = min(self.min_lane, lane)
        self.max_lane = max(self.max_lane, lane)

    def merge(self, other_cluster):
        for number, chunks in other_cluster.items():
            for chunk in chunks:
                self.add_to_lane(number, chunk)


def test_for_chunks(img, chunk_size=0.05):
    """splits the image into 5% chunks and repeats"""

    image_width = img.shape[1]
    chunk_width = int(image_width * chunk_size)
    chunks = []
    index = 0
    while index < image_width:
        chunks.append(img[:, index:min(index+chunk_width, image_width)])
        index += chunk_width

    word_layout = []
    for chunk in chunks:
        indexes = []
        empty = get_empty_lines(chunk)
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

    graph = construct_graph(word_layout)
    components = traverse_graph(graph)
    images = create_component_images(img, components, chunk_width)

    for img in images:
        cv2.imshow("test", resize_img(img))
        cv2.waitKey(0)


def create_component_images(img, components, chunk_width):
    images = []
    for component in components:
        min_lane = min(component, key=lambda x: x[0])[0]
        max_lane = max(component, key=lambda x: x[0])[0]

        x1 = int(min_lane * chunk_width)
        x2 = int((max_lane + 1) * chunk_width)
        y1 = min(component, key=lambda x: x[1])[1]
        y2 = max(component, key=lambda x: x[2])[2]

        component_img = np.ones((y2-y1, x2-x1)) * 255
        for chunk in component:
            chunk_x1 = int(chunk[0] * chunk_width)
            chunk_x2 = int((chunk[0] + 1) * chunk_width)
            chunk_y1 = chunk[1]
            chunk_y2 = chunk[2]

            local_x1 = chunk_x1 - x1
            local_x2 = chunk_x2 - x1
            local_y1 = chunk_y1 - y1
            local_y2 = chunk_y2 - y1

            component_img[local_y1:local_y2,
                          local_x1:local_x2] = img[chunk_y1:chunk_y2, chunk_x1: chunk_x2]
        images.append(component_img)
    return images


def traverse_graph(graph):
    """traverses the graph and returns sets of components"""
    components = []
    state = {node: "U" for node in graph.keys()}
    for node in state.keys():
        if state[node] == "U":
            queue = deque()
            queue.appendleft(node)
            component, state = bfs_loop(queue, graph, state, {node})
            components.append(component)
    return components


def bfs_loop(queue, graph, state, component):
    while len(queue) != 0:
        node = queue.pop()
        for v in graph[node]:
            if state[v] == "U":
                state[v] = "D"
                component.add(v)
                queue.appendleft(v)
        state[node] = "P"
    return component, state


def construct_graph(word_layout):
    """we want to construct a digraph of touching chunks.
    This should be done in a left to right manner,
    and  we should assign the graph by the tuples themselves
    as they are hashable"""
    graph = {}
    for lane in range(len(word_layout)):
        for y1, y2 in word_layout[lane]:
            touching = get_touching((lane, y1, y2), word_layout)
            for chunk in touching:
                graph[chunk] = graph.get(chunk, []) + [(lane, y1, y2)]
            graph[(lane, y1, y2)] = graph.get((lane, y1, y2), []) + touching
    return graph


def get_touching(box, word_layout):
    """returns a list of all boxes that box is touching"""
    next_lane = box[0] + 1
    if next_lane >= len(word_layout):
        return []
    results = []
    for other_box in word_layout[box[0]+1]:
        if touching(box, (next_lane, other_box[0], other_box[1])):
            results.append((next_lane, other_box[0], other_box[1]))
    return results


def touching(box1, box2):
    """takes in (lane, y1, y2) tuples, returns true if they are touching"""
    return abs(box1[0]-box2[0]) == 1 and\
        ((box1[1] <= box2[1] and box1[2] >= box2[1]) or
         (box1[2] <= box2[2] and box1[2] >= box2[1]) or
         (box2[1] <= box1[2] and box2[1] >= box1[1]) or
         (box2[2] <= box1[2] and box2[2] >= box1[1]))


def draw_word_layout(img, word_layout, chunk_width):
    for i in range(len(word_layout)):
        for word in word_layout[i]:
            img = cv2.rectangle(img, (int(i*chunk_width), word[0]),
                                (int((i+1)*chunk_width), word[1]), (255, 0, 0), 4)
    return img


def color_line(img, line_number):
    """colors a line in the given image"""
    img[line_number, :, 0] = 0
    img[line_number, :, 1] = 0

    return img


def color_lines(img, lines):
    for line in lines:
        img = color_line(img, line)
    return img


def get_empty_lines(img):
    """returns list of lines with no colored pixels"""
    result = np.zeros(img.shape[0])
    for i in range(img.shape[0]):
        result[i] = not (np.any(img[i, :] == 0))

    return result

# Word segmentation
# ========================================================


def main():
    filepath = "./detection-dataset/20240128_090136.jpg"
    img = cv2.imread(filepath)
    processed_img = preprocess_img(img)

    colored_img = test_for_chunks(processed_img)


# Character segmentation
# =========================================================
# we can start by splitting the word into connected components.
# Once this is done we can thin the image to just be 1 pixel long
# https://ieeexplore.ieee.org/document/8862590
# https://ieeexplore.ieee.org/document/9081080
# We can then take regions of only one pixel (suggesting a horizontal section),
# and then justify the segmentation from there
if __name__ == "__main__":
    main()
