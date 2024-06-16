import numpy as np

def is_bbox_within(bbox_small, bbox_large):
    """
    Author: ChatGPT

    Check if the small bounding box is within the large bounding box.
    
    Parameters:
    - bbox_small: A 4x2 array of coordinates for the small bounding box (each row is a corner).
    - bbox_large: A 4x2 array of coordinates for the large bounding box (each row is a corner).
    
    Returns:
    - True if the small bounding box is within the large bounding box, False otherwise.
    """
    # Extract the x and y coordinates
    x_small = bbox_small[:, 0]
    y_small = bbox_small[:, 1]
    x_large = bbox_large[:, 0]
    y_large = bbox_large[:, 1]

    # Find the bounds of the large bounding box
    x_min_large = np.min(x_large)
    x_max_large = np.max(x_large)
    y_min_large = np.min(y_large)
    y_max_large = np.max(y_large)

    # Check if all corners of the small bounding box are within the bounds of the large bounding box
    is_within = np.all((x_small >= x_min_large) & (x_small <= x_max_large) &
                    (y_small >= y_min_large) & (y_small <= y_max_large))
    
    return is_within

def xyxy_to_normalized_xywh(tltrblbr, image_width, image_height):
    ymin = min(tltrblbr[:,0])
    ymax = max(tltrblbr[:,0])
    xmin = min(tltrblbr[:,1])
    xmax = max(tltrblbr[:,1])
    
    width = xmax - xmin
    height = ymax - ymin
    center_x = xmin + width / 2
    center_y = ymin + height / 2

    # Normalize the coordinates
    normalized_center_x = center_x / image_width
    normalized_center_y = center_y / image_height
    normalized_width = width / image_width
    normalized_height = height / image_height

    return normalized_center_x, normalized_center_y, normalized_width, normalized_height
