import numpy as np
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from tifffile import imread
from datetime import datetime
from dataclasses import dataclass, field
from skimage.feature import multiscale_basic_features
from functools import partial
import os 

@dataclass
class RandomForestFramework(BaseFramework):

    def __init__(self, patch_path, model_name, num_classes):

        self.load_mode = LoadMode.File
        self.descriptor = "Random Forest Model"

    def generate_model_name(self, base_name="random_forest"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
        
    def create_callback(self, updater):
        self.updater = updater
    
    def train(self, num_epochs, updater=None):
        self.extract_features(image, )
    
    def predict(self, image):
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

        example_feature_params = {
            "sigma_min": 1,
            "sigma_max": 5,
            "intensity": True,
            "edges": True,
            "texture": True,
        }


        features = extract_features(image, example_feature_params)
        features.shape

