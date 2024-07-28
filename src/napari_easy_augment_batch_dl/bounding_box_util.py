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

def tltrblbr_to_normalized_xywh(tltrblbr, image_width, image_height):
    """
    Convert a bounding box defined by its top-left and bottom-right coordinates (in pixel space) 
    to a normalized bounding box format (center_x, center_y, width, height) with coordinates 
    relative to the image dimensions.

    Parameters:
    - tltrblbr (np.ndarray): Array of shape (4, 2) containing the coordinates of the bounding box 
      corners in the format [[top_left_y, top_left_x], [top_right_y, top_right_x], 
      [bottom_right_y, bottom_right_x], [bottom_left_y, bottom_left_x]].
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.

    Returns:
    - tuple: Normalized coordinates of the bounding box (normalized_center_x, normalized_center_y, 
      normalized_width, normalized_height) where all values are between 0 and 1.
    """
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

def normalized_xywh_to_tltrblbr(normalized_center_x, normalized_center_y, normalized_width, normalized_height, image_width, image_height):
    """
    Convert a normalized bounding box format (center_x, center_y, width, height) to the coordinates
    of its corners in pixel space (top-left, top-right, bottom-right, bottom-left).

    Parameters:
    - normalized_center_x (float): Normalized x-coordinate of the bounding box center (0 to 1).
    - normalized_center_y (float): Normalized y-coordinate of the bounding box center (0 to 1).
    - normalized_width (float): Normalized width of the bounding box (0 to 1).
    - normalized_height (float): Normalized height of the bounding box (0 to 1).
    - image_width (int): Width of the image in pixels.
    - image_height (int): Height of the image in pixels.

    Returns:
    - np.ndarray: Array of shape (4, 2) containing the coordinates of the bounding box corners 
      in the format [[top_left_y, top_left_x], [top_right_y, top_right_x], 
      [bottom_right_y, bottom_right_x], [bottom_left_y, bottom_left_x]].
    """
    center_x = normalized_center_x * image_width
    center_y = normalized_center_y * image_height
    width = normalized_width * image_width
    height = normalized_height * image_height

    xmin = center_x - width / 2
    ymin = center_y - height / 2
    xmax = center_x + width / 2
    ymax = center_y + height / 2

    return np.array([[ymin, xmin], [ymin, xmax], [ymax, xmax], [ymax, xmin]])