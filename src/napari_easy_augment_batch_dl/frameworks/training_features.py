import numpy as np
from skimage.feature import multiscale_basic_features


# 1.  figure out which images have labels
# 2.  make features for those images
# 3.  make vector for actual label pixels

# 4.  So who directs the extract features?  deep_learning_project ? or napari_easy_augment_batch_dl?

def extract_features(image, labels, feature_params=None):

    # loop through all slices
    for i in range(labels.shape[0]):
        if np.sum(labels[i]) > 0:
            features = extract_features_2d(image[i], feature_params)

def extract_features_2d(image, feature_params=None):
    """
    Extracts multiscale basic features from the given image using the provided or default parameters.

    Parameters:
        image (numpy.ndarray): Input image array (can be multichannel).
        feature_params (dict, optional): Dictionary containing feature extraction parameters:
            - "sigma_min": Minimum sigma for multiscale features (default: 1)
            - "sigma_max": Maximum sigma for multiscale features (default: 5)
            - "intensity": Whether to compute intensity features (default: True)
            - "edges": Whether to compute edge features (default: True)
            - "texture": Whether to compute texture features (default: True)

    Returns:
        numpy.ndarray: Array of extracted features.
    """
    if feature_params is None:
        feature_params = {
            "sigma_min": 1,
            "sigma_max": 5,
            "intensity": True,
            "edges": True,
            "texture": True,
        }

    num_channels = image.shape[-1] if len(image.shape) > 2 else 1

    for c in range(num_channels):
        features_temp = multiscale_basic_features(
            np.squeeze(image[..., c]),
            intensity=feature_params["intensity"],
            edges=feature_params["edges"],
            texture=feature_params["texture"],
            sigma_min=feature_params["sigma_min"],
            sigma_max=feature_params["sigma_max"],
            channel_axis=None,
        )
        if c == 0:
            features = features_temp
        else:
            features = np.concatenate((features, features_temp), axis=2)

    return features