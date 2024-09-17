import numpy as np
        
def pad_to_largest(images, force8bit=True):
    # Find the maximum dimensions
    max_rows = max(image.shape[0] for image in images)
    max_cols = max(image.shape[1] for image in images)
    
    # Create a list to hold the padded images
    padded_images = []
    
    for image in images:
        # Calculate the padding for each dimension
        pad_rows = max_rows - image.shape[0]
        pad_cols = max_cols - image.shape[1]
        
        if len(image.shape) == 3:
            # we occasionally hit rgba images, just use the first 3 channels
            image = image[:,:,:3]
            # Pad the array
            padded_image = np.pad(image, 
                                ((0, pad_rows), (0, pad_cols), (0,0)), 
                                mode='constant', 
                                constant_values=0)
        else:
            padded_image = np.pad(image, 
                                ((0, pad_rows), (0, pad_cols)), 
                                mode='constant', 
                                constant_values=0)
        
        if force8bit:
            min_ = np.min(padded_image)
            max_ = np.max(padded_image)
            padded_image = ((padded_image - min_) / (max_ - min_) * 255).astype(np.uint8)
        
        padded_images.append(padded_image)
    
    # Stack the padded images along a new third dimension
    result = np.array(padded_images)
    
    return result

