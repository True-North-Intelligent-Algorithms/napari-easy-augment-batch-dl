from matplotlib.pyplot import step
from napari_easy_augment_batch_dl.base_model import BaseModel, LoadMode
import numpy as np
from stardist.models import StarDist2D, Config2D
import os
from csbdeep.utils import normalize
from tnia.deeplearning.dl_helper import quantile_normalization
from tnia.deeplearning.dl_helper import collect_training_data
from tnia.deeplearning.dl_helper import divide_training_data
import keras
import json
from dataclasses import dataclass, field
from enum import Enum

class ModelType(Enum):
    STARDIST1 = "stardist-one"
    STARDIST2 = "stardist-two"

class CustomCallback(keras.callbacks.Callback):

    def __init__(self, updater, num_epochs=100):
        super(CustomCallback, self).__init__()
        self.updater = updater
        self.num_epochs = num_epochs

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        print("Start epoch {} of training; got log keys: {}".format(epoch, keys))
        if self.updater is not None:
            percent_done = int(100*epoch/self.num_epochs)
            self.updater(f"Starting epoch {epoch}", percent_done)

@dataclass
class StardistInstanceModel(BaseModel):
    prob_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})
    nms_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})
    scale: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 100.0, 'default': 1.0, 'step': 1.0})

    def __init__(self, patch_path: str = '', model_path: str = '',  num_classes: int = 1, start_model_path: str = None):
        super().__init__(patch_path, model_path, num_classes)

        self.model_path = model_path
        
        if start_model_path is None:
            self.model = None
            '''
            n_rays = 32
            axes = 'YXC'

            if axes == 'YXC':
                n_channel_in = 3
            else:
                n_channel_in = 1

            config = Config2D (n_rays=n_rays, axes=axes,n_channel_in=n_channel_in, train_patch_size = (256,256), unet_n_depth=4)
            self.model = StarDist2D(config = config, name='model_temp', basedir = model_path)
            '''
        else:
            basename = os.path.basename(start_model_path)
            basedir = os.path.dirname(start_model_path)
            self.model = StarDist2D(config = None, name=basename, basedir = basedir)

        self.custom_callback = None

        self.prob_thresh = 0.5
        self.nms_thresh = 0.4
        self.scale = 1.0
        
        self.descriptor = "Stardist Model"
        self.load_mode = LoadMode.Directory

    def create_callback(self, updater):
        self.updater = updater
        self.custom_callback = CustomCallback(updater)

    def predict(self, img: np.ndarray):
        #img_normalized = normalize(img,1,99.8, axis=(0,1))
        #img_normalized = quantile_normalization(img, quantile_low=0.001).astype(np.float32)
        img_normalized = quantile_normalization(img, quantile_low=0.003).astype(np.float32)

        labels, details =  self.model.predict_instances(img_normalized)

        return labels
    
    def train(self, num_epochs, updater=None):
       # make thread model
        '''
        for i in range(num_epochs):
            # pause
            print('epoch ',i)
            time.sleep(1)
            
            if self.updater is not None:
                self.updater(f"Starting epoch {i}", i)
                #print(f"Starting epoch {i}")
                # do something

        '''
        json_name = os.path.join(self.patch_path, 'info.json')

        self.num_epochs = num_epochs

        if os.path.exists(json_name):
            with open(json_name, 'r') as f:
                data = json.load(f)
                axes = data['axes']
                if axes == 'YXC':
                    n_channel_in = 3
                    add_trivial_channel = False
                else:
                    n_channel_in = 1
                    add_trivial_channel = True
            
            model_name = self.generate_model_name('stardist')

            # if model is None create one            
            if self.model is None:
                config = Config2D (n_rays=32, axes=axes, n_channel_in=n_channel_in, train_patch_size = (256,256), unet_n_depth=3)
                self.model = StarDist2D(config = config, name=model_name, basedir = self.model_path)
            
            X, Y = collect_training_data(self.patch_path, sub_sample=1, downsample=False, normalize_input=False, add_trivial_channel = add_trivial_channel)
            X_train, Y_train, X_val, Y_val = divide_training_data(X, Y, val_size=2)
            
            self.model.prepare_for_training()
            
            if self.custom_callback is not None:
                self.custom_callback.num_epochs = num_epochs
                self.model.callbacks.append(self.custom_callback)
            
            self.model.train(X_train, Y_train, validation_data=(X_val,Y_val),epochs=num_epochs, steps_per_epoch=100)

    @property
    def model_type(self):
        return self._model_type

    def get_model_names(self):
        return ['model1', 'model2']


    def load_model_from_disk(self, full_model_name):
        # get path and name from model_name
        model_path = os.path.dirname(full_model_name)
        model_name = os.path.basename(full_model_name)

        self.model = StarDist2D(config=None, name=model_name, basedir=model_path)