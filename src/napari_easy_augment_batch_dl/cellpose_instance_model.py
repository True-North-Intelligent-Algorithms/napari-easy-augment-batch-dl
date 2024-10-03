from napari_easy_augment_batch_dl.base_model import BaseModel, LoadMode
import numpy as np
from tnia.deeplearning.dl_helper import collect_training_data
from cellpose import models, io
from dataclasses import dataclass, field

@dataclass
class CellPoseInstanceModel(BaseModel):
    
    diameter: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 500.0, 'default': 30.0, 'step': 1.0})
    prob_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1})
    flow_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1})
    chan_segment: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100, 'default': 0, 'step': 1})
    chan2: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100, 'default': 1, 'step': 1})

    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model: str = None):
        super().__init__(patch_path, model_path, num_classes)

        # start logger (to see training across epochs)
        logger = io.logger_setup()

        # if no model set to none and wait until user selects a model 
        if start_model is None:
            self.model = None
        # if model passed in set it
        elif type(start_model) == models.Cellpose:
            self.model = start_model
        # otherwise if path passed in load model
        else:
            self.model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=start_model)

        self.diameter = 30
        self.prob_thresh = 0.0
        self.flow_thresh = 0.4
        self.chan_segment = 1
        self.chan2 = 0

        self.descriptor = "CellPose Instance Model"
        self.load_mode = LoadMode.File

    def load_model_from_disk(self, model_path):
        self.model = models.CellposeModel(gpu=True, model_type=None, pretrained_model=model_path)

    def train(self, num_epochs, updater=None):
        add_trivial_channel = False
        X, Y = collect_training_data(self.patch_path, sub_sample=1, downsample=False, normalize_input=False, add_trivial_channel = add_trivial_channel, relabel=True)

        train_percentage = 0.9

        X_ = X.copy()
        Y_ = Y.copy()

        X_train = X_[:int(len(X_)*train_percentage)]
        Y_train = Y_[:int(len(Y_)*train_percentage)]
        X_test = X_[int(len(X_)*train_percentage):]
        Y_test = Y_[int(len(Y_)*train_percentage):]

        print(X_train[0].shape)
        print(Y_train[0].shape)

        #print(help(self.model.train_seg))

        if self.model is None:
            self.model = models.CellposeModel(gpu=True, model_type=None)

        from cellpose import train

        model_name = self.generate_model_name('cellpose')
      
        new_model_path = train.train_seg(self.model.net, X_train, Y_train, 
            test_data=X_test,
            test_labels=Y_test,
            channels=[self.chan_segment, self.chan2], 
            save_path=self.model_path, 
            n_epochs = num_epochs,
            # TODO: make below GUI options
            #learning_rate=learning_rate, 
            #weight_decay=weight_decay, 
            #nimg_per_epoch=200,
            model_name=model_name)

    def predict(self, img: np.ndarray):
        return self.model.eval(img, channels=[self.chan_segment, self.chan2], diameter = self.diameter, flow_threshold=self.flow_thresh, cellprob_threshold=self.prob_thresh)[0]

    def get_model_names(self):
        return ['notset', 'cyto3', 'tissuenet_cp3']
    
    def set_pretrained_model(self, model_name):
        if model_name != 'notset':
            if model_name in self.get_model_names():
                self.model = models.CellposeModel(gpu=True, model_type=model_name)
