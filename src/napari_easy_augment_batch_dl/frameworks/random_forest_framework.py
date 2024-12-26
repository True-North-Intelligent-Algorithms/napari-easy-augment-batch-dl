import numpy as np
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from tifffile import imread
from datetime import datetime
from dataclasses import dataclass, field
from skimage.feature import multiscale_basic_features
from functools import partial
import os 
from napari_easy_augment_batch_dl.zarr_helper import get_zarr_store
@dataclass
class RandomForestFramework(BaseFramework):

    default_feature_params = {
        "sigma_min": 1,
        "sigma_max": 5,
        "intensity": True,
        "edges": True,
        "texture": True,
    }

    def __init__(self, parent_path, num_classes):
        super().__init__(parent_path, num_classes)

        self.ml_labels_path = os.path.join(parent_path, "ml", "ml_labels")

        self.load_mode = LoadMode.File
        self.descriptor = "Random Forest Model"

    def generate_model_name(self, base_name="random_forest"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
        
    def create_callback(self, updater):
        self.updater = updater
    
    def train(self, num_epochs, updater=None):
        updater('time to train the random forest')
        store = get_zarr_store(self.ml_labels_path)
        images = store['images']

        for i in range(images.shape[0]):
            image = images[i]

            if image.sum() > 0:
                updater(f"image {i} has sum {image.sum()}")

    def predict(self, image):
        print('time to predict with the random forest')
        pass
    
    def load_model_from_disk(self, model_name):
        pass

    def extract_features(image, feature_params):
        features_func = partial(
            multiscale_basic_features,
            intensity=feature_params["intensity"],
            edges=feature_params["edges"],
            texture=feature_params["texture"],
            sigma_min=feature_params["sigma_min"],
            sigma_max=feature_params["sigma_max"],
            channel_axis=None,
        )
        # print(f"image shape {image.shape} feature params {feature_params}")
        
        for c in range(image.shape[-1]):
            features_temp = features_func(np.squeeze(image[..., c]))
            if c == 0:
                features = features_temp
            else:
                features = np.concatenate((features, features_temp), axis=2)
        #features = features_func(np.squeeze(image))
        
        return features

        features = extract_features(image, example_feature_params)
        features.shape

