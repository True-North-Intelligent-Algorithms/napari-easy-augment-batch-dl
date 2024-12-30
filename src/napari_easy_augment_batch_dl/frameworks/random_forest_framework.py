import numpy as np
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode, TrainMode
from tifffile import imread
from datetime import datetime
from dataclasses import dataclass, field
from skimage.feature import multiscale_basic_features
from functools import partial
import os 
from napari_easy_augment_batch_dl.zarr_helper import get_zarr_store
from napari_easy_augment_batch_dl.utility import collect_all_images
from napari_easy_augment_batch_dl.frameworks.training_features import TrainingFeatures
from sklearn.ensemble import RandomForestClassifier
from skimage import future
import dask.array as da

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
        self.ml_features_path = os.path.join(parent_path, "ml", "ml_features")

        self.load_mode = LoadMode.File
        self.descriptor = "Random Forest Model"

        self.train_mode = TrainMode.Pixels

        self.clf = RandomForestClassifier(
            n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)


    def generate_model_name(self, base_name="random_forest"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
    
    def create_features(self, images, labels, features):
        self.images = images
        self.labels = labels
        self.features = features
        self.training_features = TrainingFeatures(self.images, self.labels, self.features)

        #self.dask_image = da.from_zarr(self.images)
        self.dask_labels = da.from_zarr(self.labels)
        self.dask_features = da.from_zarr(self.features)
        
    def create_callback(self, updater):
        self.updater = updater
    
    def train(self, num_epochs, updater=None):

        updater('time to train the random forest')

        updater('image size is {}'.format(self.images.shape))
        updater('feature size is {}'.format(self.features.shape))

        for i in range(self.images.shape[0]):
            image = self.images[i]
            label = self.labels[i]

            if label.sum() > 0:
                updater(f"image {i} has shape {image.shape}")
                updater(f"labels {i} has sum {label.sum()}")
                
                if self.features[i,:,:,:].sum() == 0:
                    updater(f"extracting features for image {i}")
                    self.features[i,:,:,:] = self.extract_features(image, self.default_feature_params)
                else:
                    updater(f"features {i} already exist")

        '''
        mask = self.dask_labels>0
        indices = da.nonzero(mask.flatten())
        indices=indices + (slice(None),)

        features_ = self.dask_features[indices]
        # We shift labels - 1 because background is 0 and has special meaning, but models need to start at 0
        labels_ = self.dask_labels[self.dask_labels > 0] - 1
        '''
        mask = self.labels[:] > 0
        features_ = self.features[:]
        features_ = features_[mask]

        labels_ = self.labels[:]
        labels_ = labels_[mask] - 1

        self.clf.fit(features_, labels_)

    def predict(self, image):
        print('time to predict with the random forest')
        features = self.extract_features(image, self.default_feature_params)
        model = self.clf
        prediction = future.predict_segmenter(features.reshape(-1, features.shape[-1]), model).reshape(features.shape[:-1]) + 1

        return np.transpose(prediction)
        
        pass
    
    def load_model_from_disk(self, model_name):
        pass

    def extract_features(self, image, feature_params, updater=None):
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
            if updater is not None:
                updater(f"extracting features for channel {c}")
            
            features_temp = features_func(np.squeeze(image[..., c]))
            if c == 0:
                features = features_temp
            else:
                features = np.concatenate((features, features_temp), axis=2)
        
        return features


