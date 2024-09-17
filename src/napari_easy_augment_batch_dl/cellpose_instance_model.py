from napari_easy_augment_batch_dl.base_model import BaseModel
import numpy as np
import os
from tnia.deeplearning.dl_helper import collect_training_data
from cellpose import models, io

class CellPoseInstanceModel(BaseModel):

    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model: str = None):
        super().__init__(patch_path, model_path, num_classes)

        # start logger (to see training across epochs)
        logger = io.logger_setup()
        
        if start_model is None:
            # DEFINE CELLPOSE MODEL (without size model)
            self.model = models.CellposeModel(gpu=True, model_type=None)
        elif type(start_model) == models.Cellpose:
            self.model = start_model
        else:
            self.model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=start_model)

        self.diameter = 30
        self.flow_threshold = 0.4
        self.cellprob_threshold = 0.0
    
    def create_callback(self, updater):
        pass

    def train(self, num_epochs, updater=None):
        add_trivial_channel = False
        X, Y = collect_training_data(self.patch_path, sub_sample=1, downsample=False, normalize_input=False, add_trivial_channel = add_trivial_channel, relabel=True)

        train_percentage = 0.9

        X_ = X.copy()
        Y_ = Y.copy()

        X_train = X_[:int(len(X_)*0.8)]
        Y_train = Y_[:int(len(Y_)*0.8)]
        X_test = X_[int(len(X_)*0.8):]
        Y_test = Y_[int(len(Y_)*0.8):]

        print(X_train[0].shape)
        print(Y_train[0].shape)

        #print(help(self.model.train_seg))

        from cellpose import train
      
        new_model_path = train.train_seg(self.model.net, X_train, Y_train, 
            #test_data=X_val,
            #test_labels=Y_val,
            channels=[0,1], 
            save_path=self.model_path, 
            n_epochs = num_epochs,
            #learning_rate=learning_rate, 
            #weight_decay=weight_decay, 
            #nimg_per_epoch=200,
            model_name='cellpose_thirdtry')

    def predict(self, img: np.ndarray):
        return self.model.eval(img, channels=[0, 1], diameter = self.diameter, flow_threshold=self.flow_threshold, cellprob_threshold=self.cellprob_threshold)[0]
