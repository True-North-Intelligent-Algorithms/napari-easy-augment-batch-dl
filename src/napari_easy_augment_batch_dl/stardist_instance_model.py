from napari_easy_augment_batch_dl.base_model import BaseModel
import numpy as np
from stardist.models import StarDist2D, Config2D
import os
from csbdeep.utils import normalize
from tnia.deeplearning.dl_helper import quantile_normalization
from tnia.deeplearning.dl_helper import collect_training_data
from tnia.deeplearning.dl_helper import divide_training_data
import keras

class CustomCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

class StardistInstanceModel(BaseModel):
    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model_path: str = None):
        super().__init__(patch_path, model_path, num_classes)

        
        if start_model_path is None:
            n_rays = 32
            axes = 'YXC'

            if axes == 'YXC':
                n_channel_in = 3
            else:
                n_channel_in = 1

            config = Config2D (n_rays=n_rays, axes=axes,n_channel_in=n_channel_in, train_patch_size = (256,256), unet_n_depth=4)

            self.model = StarDist2D(config = config, name='model_temp', basedir = model_path)
        else:
            basename = os.path.basename(start_model_path)
            basedir = os.path.dirname(start_model_path)
            self.model = StarDist2D(config = None, name=basename, basedir = basedir)

    def predict(self, img: np.ndarray):
        #img_normalized = normalize(img,1,99.8, axis=(0,1))
        img_normalized = quantile_normalization(img, quantile_low=0.001).astype(np.float32)

        labels, details =  self.model.predict_instances(img_normalized)

        return labels
    
    def train(self, num_epochs, updater=None):
       add_trivial_channel = False
       X, Y = collect_training_data(self.patch_path, sub_sample=1, downsample=False, normalize_input=False, add_trivial_channel = add_trivial_channel)
       X_train, Y_train, X_val, Y_val = divide_training_data(X, Y, val_size=2)
       self.model.prepare_for_training()
       self.model.callbacks.append(CustomCallback())
       self.model.train(X_train, Y_train, validation_data=(X_val,Y_val),epochs=num_epochs, steps_per_epoch=100)
       #model.train(X_) 