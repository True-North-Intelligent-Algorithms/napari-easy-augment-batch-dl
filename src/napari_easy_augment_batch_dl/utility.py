import numpy as np
        
def pad_to_largest(images, force8bit=False):

    # TODO: this is actually a pretty complicated function, not only do we pad but 
    # we also normalize the images to 8 bit, and we also convert to rgb if the image is not 3 channel
    # will need continued work and refactoring, as we are essentially doing multi-time, multi-channel, multi-format 
    # conversion for display. 
    
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
 
        if (len(padded_image.shape) > 2):

            if padded_image.shape[2] != 3:
                padded_image = multi_channel_to_rgb(padded_image)
       
        if force8bit:
            #min_ = np.min(padded_image)
            #max_ = np.max(padded_image)
            #padded_image = ((padded_image - min_) / (max_ - min_) * 255).astype(np.uint8)

            if (len(padded_image.shape) > 2):
                padded_image = normalize_per_channel(padded_image)
            else:
                min_ = np.min(padded_image)
                max_ = np.max(padded_image)
                padded_image = ((padded_image - min_) / (max_ - min_) * 255).astype(np.uint8)

        padded_images.append(padded_image)

    shapes = [image.shape for image in padded_images]

    # BN was toying with the idea of displaying 3 channel and 1 channel images together but it is a bit messy, so commented out for now
    '''
    if len(set(shapes)) > 1:
        for i in range(len(padded_images)):
            if len(padded_images[i].shape)==2:
                padded_images[i] = padded_images[i][:,:,np.newaxis]
                padded_images[i] = multi_channel_to_rgb(padded_images[i])
    '''
    # final check to see if all images have the same number of dimensions
    
    # Stack the padded images along a new third dimension
    result = np.array(padded_images)

    if force8bit:
        result = result.astype(np.uint8)
    
    return result

def normalize_per_channel(padded_image):
    normalized_image = np.zeros_like(padded_image, dtype=np.uint8)
    for c in range(padded_image.shape[2]):  # Assuming the last dimension is the channel
        channel = padded_image[:, :, c]
        min_ = channel.min()
        max_ = channel.max()
        normalized_image[:, :, c] = ((channel - min_) / (max_ - min_) * 255).astype(np.uint8)
    return normalized_image

def multi_channel_to_rgb(im):

    # if too many channels remove the last ones
    if im.shape[2] > 3:
        im_ = im[:,:,:3]
    if im.shape[2]<3:
        im_ = np.concatenate([im, np.zeros(im.shape[:2])[:, :, None]], axis=2)
    if im.shape[2]<3:
        im_ = np.concatenate([im_, np.zeros(im.shape[:2])[:, :, None]], axis=2)
    return im_

def unpad_to_original(images, original_images):
    unpadded_images = []        
    for n in range(len(images)):
        height, width = original_images[n].shape[:2]

        image = images[n][:height, :width]
        unpadded_images.append(image)

    return unpadded_images

