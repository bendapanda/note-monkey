# https://www.researchgate.net/publication/220888490_Unconstrained_Handwritten_Text-line_Segmentation_Using_Morphological_Operation_and_Thinning_Algorithm
import numpy as np
import cv2
import preprocessor
from sklearn.neighbors import KernelDensity
from scipy.signal import argrelextrema
import os
from copy import deepcopy
import matplotlib.pyplot as plt

class Line():
    def __init__(self, sections):
        """Takes a list of boxes with percentages as points"""
        self.sections = np.array(sections)
    
    def segment_image(self, image: np.ndarray):
        """Takes in any image, and performs its line segmentation on it."""
        height, width = image.shape[:2]
        min_x = int(width * np.min(self.sections[:, 0]))
        max_x = int(width * np.max(self.sections[:, 2]))
        min_y = int(height * np.min(self.sections[:, 1]))
        max_y = int(height * np.max(self.sections[:, 3]))

        if len(image.shape) <= 2:
            line_image = np.ones((max_y-min_y, max_x-min_x))
        else:
            line_image = np.ones((max_y-min_y, max_x-min_x, image.shape[2]))
        line_image *= 255
        line_image = line_image.astype(np.uint8)
        for x1, y1, x2, y2 in self.sections:
            
            line_image[int(height*y1)-min_y: int(height*y2)-min_y,
                       int(width*x1)-min_x:int(width*x2)-min_x] = image[int(height*y1):int(height*y2), int(width*x1):int(width*x2)]
        return line_image

def segment(image, verbosity=0) -> list[Line]:
    if verbosity >= 2:
        cv2.imshow("original image", preprocessor.resize_img(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if verbosity >= 2:
        cv2.imshow("preprocessed image", preprocessor.resize_img(processed))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    # Here we do two passes of essentially the same data. The first pass is to find the best
    # stripe spacing, and the second is to process with that spacing
    image_stripes, chunk_width = split_image(processed, 0.2)
    painted_stripes = []
    for stripe in image_stripes:
        stripe = paint_stripe(stripe)
        stripe = stripe.astype(np.uint8)
        stripe = preprocessor.otsu_thresholding(stripe)
        painted_stripes.append(stripe)
    
    if verbosity >= 2:
        cv2.imshow("binarized image", preprocessor.resize_img(np.hstack(painted_stripes)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    painted_stripes = smooth_stripes(painted_stripes, verbosity=verbosity)
    if verbosity >= 2:
        cv2.imshow("smoothed", preprocessor.resize_img(np.hstack(painted_stripes)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    painted_stripes = new_dialation_operation(painted_stripes)
    if verbosity >= 2:
        cv2.imshow("dialated", preprocessor.resize_img(np.hstack(painted_stripes)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    painted_image = draw_segments(painted_stripes, image)
    
    resized = preprocessor.resize_img(painted_image)
    if verbosity >= 1:
        cv2.imshow("painting test", resized.astype(np.uint8))
        cv2.waitKey(0)
        cv2.destroyAllWindows()    
    
    lines = create_lines(painted_stripes)
    save_name = filename.split('.')[0]
    save_name = save_name.split('/')[1]
    save_count = 0
    for line in lines:
        # save_dir = "/home/bensh/Documents/code/note-monkey/line-images"
        # save_filename = f"{save_name}_{save_count}.jpg"
        # print(save_filename)
        # save_count += 1
        # cv2.imwrite(os.path.join(save_dir, save_filename), line.segment_image(preprocessor.otsu_thresholding(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))))
        if verbosity >= 1:
            cv2.imshow("line", preprocessor.resize_img(line.segment_image(image)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return lines

def create_lines(stripes):
    lines = []

    boxes = []
    for i in range(len(stripes)):
        boxes.append(get_boxes(stripes[i], 0))
    
    stripe_index = 0
    box_index = 0
    while not all(len(box_list) == 0 for box_list in boxes):
        if len(boxes[stripe_index]) != 0:
            # add image to pieces
            line_image_coords = pass_over_line(boxes, stripes, stripe_index, box_index, current_image_pieces=[])
            line_coords_percentages = []
            width = np.hstack(stripes).shape[1]
            height = np.hstack(stripes).shape[0]
            for x1, y1, x2, y2 in line_image_coords:
                line_coords_percentages.append((x1/width, y1/height, x2/width, y2/height))
            
            lines.append(Line(line_coords_percentages))
                
        else:
            stripe_index += 1

    return lines

    

def create_segment_images(stripes, image):
    """given a list of stripes, segments the image into
    line images."""
    # create boxes. 
    # go through each box in the list.
    # add box segment to the array
    # look to the right to see if it is touching anything
    # if it is, delete the left box, and move to that box
    # repeat until we can't
    # go back to start of the list and repeat
    line_images = []

    boxes = []
    for i in range(len(stripes)):
        boxes.append(get_boxes(stripes[i], 0))
    
    stripe_index = 0
    box_index = 0
    while not all(len(box_list) == 0 for box_list in boxes):
        if len(boxes[stripe_index]) != 0:
            # add image to pieces
            line_image_coords = pass_over_line(boxes, stripes, stripe_index, box_index, current_image_pieces=[])
            line_image = create_line_image(line_image_coords, image)
            line_images.append(line_image)
        else:
            stripe_index += 1

    return line_images


def create_line_image(coords, image):
    max_x2 = max(coords, key=lambda x:x[2])[2]
    max_y2 = max(coords, key=lambda x: x[3])[3]
    min_x1 = min(coords, key=lambda x: x[0])[0]
    min_y1 = min(coords, key=lambda x: x[1])[1]

    line_image = np.ones((max_y2-min_y1, max_x2-min_x1, 3)) * 255
    for x1, y1, x2, y2 in coords:
        line_image[y1-min_y1:y2-min_y1, x1-min_x1:x2-min_x1] = image[y1:y2, x1:x2]
    return line_image.astype(np.uint8)

def pass_over_line(boxes, stripes, current_index, current_box_index, current_image_pieces=[]):
    current_box = boxes[current_index][current_box_index]
    current_image_pieces.append(get_box_image_coords(current_box, current_index, stripes))
    

    boxes[current_index].pop(current_box_index)
    current_index += 1
    if current_index < len(boxes):
        new_box_index = None
        for i in range(len(boxes[current_index])):
            if touching(current_box, boxes[current_index][i]):
                new_box_index = i
        if new_box_index is None:
            return current_image_pieces
        else:
            return pass_over_line(boxes,stripes, current_index, new_box_index, current_image_pieces=current_image_pieces)
    return current_image_pieces


def get_box_image_coords(box, x_index, stripes):
    stripe_width = stripes[0].shape[1]
    boxes_in_stripe = get_boxes(stripes[x_index], 0)
    box_index = boxes_in_stripe.index(box)
    if box_index == 0:
        y1 = 0
    else: 
        y1 = int((boxes_in_stripe[box_index-1][1] + box[0]) / 2)

    if box_index == len(boxes_in_stripe) - 1:
        y2 = stripes[x_index].shape[0]
    else:
        y2 = int((boxes_in_stripe[box_index + 1][0] + box[1]) / 2)

    x1 = int(stripe_width * x_index)
    if x_index == len(stripes) -1:
        x2 = int(stripe_width * x_index) + stripes[-1].shape[1]
    else:
        x2 = int(stripe_width * (x_index + 1))

    return (x1, y1, x2, y2)


def draw_segments(stripes, original_img):
    result = original_img.copy()
    stripe_width = stripes[0].shape[1]
    for i in range(len(stripes)):
        boxes = get_boxes(stripes[i], 255)
        for box in boxes:
            height = int((box[0] + box[1]) / 2)
            result[height-3:height+3, int(i*stripe_width):int((i+1)*stripe_width)] = (255, 0, 0)
    return result 

def remove_large_black_boxes(stripes):
    stripes = deepcopy(stripes)
    all_black_heights = []
    all_black_boxes = []
    for stripe in stripes:
        black_boxes = get_boxes(stripe, 0)
        black_heights = [box[1]-box[0] for box in black_boxes]
        all_black_boxes.append(black_boxes)
        all_black_heights += black_heights
    all_black_heights = np.array(all_black_heights)

    avg_height = np.mean(all_black_heights)
    for i in range(len(stripes)):
        for box in all_black_boxes[i]: 
            if box[1]-box[0] >= 2*avg_height:
                stripes[i][box[0]:box[1]+1, :] = 255
    return stripes

def dialation_operation(stripes, dialation_constant = 3):
    
    stripes = remove_large_black_boxes(stripes)

    # Now to perform structuring on the stripes
    new_stripes = deepcopy(stripes)
    for i in range(len(stripes)):
        for j in range(stripes[i].shape[0]):
            if stripes[i][j][0] == 0:
                for k in range(max(0, i-dialation_constant), min(len(stripes), i+dialation_constant)):
                    new_stripes[k][j, :] = 0
            

    return new_stripes

def new_dialation_operation(stripes, connection_distance=2):
    """when we dialate boxes, we need to do two things:
    First, make sure that horizontal cavities are filled,
    so holes in chunks, or divots in the horizontal direction.
    
    Then, we need to connect parts that should be connected. To do this, identify
    left/right edges, and test to see if there are any intersections within
    say 2 stripes of that point"""
    # remove large black chunks
    stripes = remove_large_black_boxes(stripes)
    stripes = fill_horizontal_cavities(stripes)
    stripes = connect_disconnected_chunks(stripes, connection_distance)
    stripes = extend_edge_chunks(stripes, extension_distance=connection_distance)

    return stripes

def extend_edge_chunks(stripes, extension_distance=1):
    """It is beneficial to extend lines that finish before the
    end of the image. So, we locate boxes that are isolated on the left and right,
    and then so long as they are not connected by the extension, we make it."""
    boxes = []
    for i in range(len(stripes)):
        boxes.append(get_boxes(stripes[i], 0))
    
    #TODO allow multiple extensions, not just one
    for stripe_index in range(len(boxes)):
        for box_index in range(len(boxes[stripe_index])):

            isolated_left = True
            if stripe_index > 0:
                for other_box in boxes[stripe_index-1]:
                    if touching(other_box, boxes[stripe_index][box_index]):
                        isolated_left = False
            else:
                isolated_left = False
            
            isolated_right = True
            if stripe_index < len(boxes) - 1:
                for other_box in boxes[stripe_index+1]:
                    if touching(other_box, boxes[stripe_index][box_index]):
                        isolated_right  = False
            else:
                isolated_right = False

            if isolated_left:
                point_to_extend = int((boxes[stripe_index][box_index][1] + boxes[stripe_index][box_index][0]) / 2)
                stripes[stripe_index-1][point_to_extend-1:point_to_extend+1, :] = 0
            if isolated_right:
                point_to_extend = int((boxes[stripe_index][box_index][1] + boxes[stripe_index][box_index][0]) / 2)
                stripes[stripe_index+1][point_to_extend-1:point_to_extend+1, :] = 0
    return stripes



def connect_disconnected_chunks(stripes, connection_distance):
     # Now, connect disconnected chunks
    #loop over all stripes and boxes
    # if box is disconnected on one side, look at the next one.
    # if it could be connected there, connect them
    new_stripes = deepcopy(stripes)
    boxes = []
    for i in range(len(new_stripes)):
        boxes.append(get_boxes(new_stripes[i], 0))
    for stripe_index in range(len(new_stripes)):
        for box_index in range(len(boxes[stripe_index])):
            left_rows_to_check = []
            right_rows_to_check = []
            for offset in range(1, connection_distance+1):
                if stripe_index - offset >= 0:
                    left_rows_to_check.append(boxes[stripe_index - offset])
                if stripe_index + offset <= len(boxes) - 1:
                    right_rows_to_check.append(boxes[stripe_index + offset])

            left_index = 0
            connection_made = False
            left_box_to_connect = None
            while not connection_made and left_index < len(left_rows_to_check):
                for other_box in left_rows_to_check[left_index]:
                    if touching(other_box, boxes[stripe_index][box_index]):
                        connection_made = True
                        left_box_to_connect = (stripe_index-left_index-1, other_box[0], other_box[1])
                left_index += 1
            
            right_index = 0
            connection_made = False
            right_box_to_connect = None
            while not connection_made and right_index < len(right_rows_to_check):
                for other_box in right_rows_to_check[right_index]:
                    if touching(other_box, boxes[stripe_index][box_index]):
                        
                        connection_made = True
                        right_box_to_connect = (stripe_index+right_index+1, other_box[0], other_box[1])
                right_index += 1

            # Now, connect left and right boxes to connect if they are not already connected
            if left_box_to_connect is not None and abs(stripe_index - left_box_to_connect[0]) > 1:
                start_point_to_draw_through = int(max(left_box_to_connect[1], boxes[stripe_index][box_index][0]))
                end_point_to_draw_through = int(min(left_box_to_connect[2], boxes[stripe_index][box_index][1]))
                for offset in range(1, abs(stripe_index - left_box_to_connect[0])):
                    new_stripes[stripe_index - offset][start_point_to_draw_through:end_point_to_draw_through, :] = 0

            if right_box_to_connect is not None and abs(stripe_index - right_box_to_connect[0]) > 1:
                start_point_to_draw_through = int((max(right_box_to_connect[1], boxes[stripe_index][box_index][0])))
                end_point_to_draw_through = int(min(right_box_to_connect[2], boxes[stripe_index][box_index][1]))
                for offset in range(1, abs(stripe_index + right_box_to_connect[0])):
                    new_stripes[stripe_index - offset][start_point_to_draw_through:end_point_to_draw_through, :] = 0
    
    return new_stripes


def fill_horizontal_cavities(stripes):
    """takes in a list of stripes, and fills in any horizontal overhangs"""
    #First, fill cavities:
    new_stripes = deepcopy(stripes)
    old_stripes = [np.zeros_like(new_stripes[i] for i in range(len(new_stripes)))]
    while any(np.any(new_stripes[i] != old_stripes[i]) for i in range(len(new_stripes))):
        old_stripes = new_stripes
        new_stripes = deepcopy(new_stripes)
        # go through each stripe and find pairs
        boxes = []
        for i in range(len(new_stripes)):
            boxes.append(get_boxes(new_stripes[i], 0))

        for stripe_index in range(len(new_stripes)):          
            for box_index in range(len(boxes[stripe_index]) - 1):
                top_box = boxes[stripe_index][box_index]
                bottom_box = boxes[stripe_index][box_index+1]

                boxes_to_consider = []
                if stripe_index > 0:
                    boxes_to_consider += boxes[stripe_index - 1]
                if stripe_index < len(new_stripes) - 1:
                    boxes_to_consider += boxes[stripe_index+1]

                boxes_connected = False
                for other_box in boxes_to_consider:
                    if touching(other_box, top_box) and touching(other_box, bottom_box):
                        boxes_connected = True
                
                if boxes_connected:
                    new_stripes[stripe_index][top_box[1]:bottom_box[0], :] = 0
    return new_stripes


def get_white_box_mode(stripes):
    """takes in a list of stripes and returns the mode
    of the white boxes using an estimation"""
    all_white_boxes = []
    all_white_heights = []

    for stripe in stripes:  
        white_boxes = get_boxes(stripe, 255)

        white_heights = [box[1] - box[0] for box in white_boxes]
        white_heights = np.array(white_heights)

        all_white_heights.append(white_heights)
        all_white_boxes.append(white_boxes)

    all_white_heights = np.concatenate(all_white_heights)
    all_white_heights = all_white_heights.reshape(-1, 1)
    kde = KernelDensity(bandwidth=0.1).fit(all_white_heights)
    x = np.linspace(0, np.max(all_white_heights), 300)
    y = kde.score_samples(x.reshape(-1, 1))

    mode = np.argmax(y)
    return x[mode]

def get_boxes(stripe: np.ndarray, colour:int):
    """returns start and stop coordinates of all the boxes of that colour
    in the stripe image"""
    if colour not in [255, 0]:
        raise ValueError("colour can only be black or white")

    stripe_boxes = []
    started = None
    for i in range(stripe.shape[0]):
        if stripe[i][0] == colour and started is None:
            started = i
        elif stripe[i][0] != colour and started is not None:
            stripe_boxes.append((started, i-1))
            started = None
    if started is not None:
        stripe_boxes.append((started, stripe.shape[0]))
    
    return stripe_boxes

def smooth_stripes(stripes: list[np.ndarray], verbosity=0):
    """smoothes the stripe to eliminate random
     black/white rectangles"""
    # First get heights of white rectangles

    all_white_boxes = []
    all_black_boxes = []
    all_white_heights = []
    all_black_heights = []

    for stripe in stripes:  
        white_boxes = get_boxes(stripe, 255)

        white_heights = [box[1] - box[0] for box in white_boxes]
        white_heights = np.array(white_heights)

        all_white_heights.append(white_heights)
        all_white_boxes.append(white_boxes)


    #removing small while sections
    #TODO test mode on this
    if verbosity>=4:
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
    for i in range(len(stripes)):
        for box in all_white_boxes[i]:
            if box[1]-box[0] < global_median*MEADIAN_SCALING_FACTOR:
                # fill box black
                stripes[i][box[0]:box[1]+1, :] = 0
    
    if verbosity>=3:
        cv2.imshow("white chunks removed", preprocessor.resize_img(np.hstack(stripes)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    for stripe in stripes:  
        black_boxes = get_boxes(stripe, 0)

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
    
    for i in range(len(stripes)):
        for box in all_black_boxes[i]:
            if box[1]-box[0] < 0.5*T:
                stripes[i][box[0]:box[1]+1, :] = 255
    
    return stripes

def touching(box1, box2):
        """takes in (lane, y1, y2) tuples, returns true if they are touching"""
        return ((box1[0] <= box2[0] and box1[0] >= box2[0]) or
             (box1[1] <= box2[1] and box1[1] >= box2[0]) or
             (box2[0] <= box1[1] and box2[0] >= box1[0]) or
             (box2[1] <= box1[1] and box2[1] >= box1[0]))



def paint_stripe(stripe):
    painted_stripe = np.ones(stripe.shape)
    for i in range(stripe.shape[0]):
        stripe_color = np.sum(stripe[i, :]) / len(stripe[i, :])
        painted_stripe[i, :] *= stripe_color
    return painted_stripe
    

         

def split_image(image: np.ndarray, chunk_percentage) -> list[np.ndarray]:
        """splits an image into verticle columns of
        chunk_percentage"""
        width = image.shape[1]
        chunk_width = int(width * chunk_percentage)
        chunks = []
        index = 0
        while index+chunk_width < width:
            chunks.append(image[:, index:index+chunk_width])
            index += chunk_width
        chunks.append(image[:, index:])

        return chunks, chunk_width
        
if __name__ == "__main__":
    for filename in os.listdir("detection-dataset"):
        segment("detection-dataset/" + filename, verbosity=0)