from sympy import im
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from dataclasses import dataclass, field
import os
import numpy as np
import torch
from torch_em.data import MinInstanceSampler
import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
import imageio
import glob
import shutil
import tempfile 
from PyQt5.QtCore import QObject, pyqtSignal

class ProgressSignals(QObject):
    pbar_total = pyqtSignal(int)
    pbar_update = pyqtSignal(int)
    pbar_description = pyqtSignal(str)


class CustomCallback():

    def __init__(self, updater, num_epochs=100):
        self.updater = updater
        self.num_epochs = num_epochs
        self.total_steps = 500  

    def on_train_begin(self):
        print("Starting training")

    def on_train_end(self):
        print("Stop training")

    def set_total(self, total):
        
        if self.updater is not None:
            self.updater("Starting microsam training", 0)
            self.total_steps = total
            self.steps_done = 0

    def on_epoch_update(self, step):
        if self.updater is not None:
            self.steps_done = self.steps_done+step
            percent_done = int(100*self.steps_done/self.total_steps)
            if self.steps_done % 10 == 0:
                self.updater(f"{self.steps_done} steps done of {self.total_steps}")

@dataclass
class MicroSamInstanceFramework(BaseFramework):
    """
    Micro-sam Instance Framework

    This framework is used to train a Microsam Instance Segmentation model.
    """
    
    # below are the parameters that are harvested for automatic GUI generation

    # first set of parameters have advanced False and training False and will be shown in the main dialog
    tile_size: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 64, 'max': 2048, 'default': 384, 'step': 1})
    halo_size: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 8, 'max': 2048, 'default': 64, 'step': 1})
    prediction_channel: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 10, 'default': 1, 'step': 1})
    
    foreground_threshold: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': True, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})
    center_distance_threshold: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': True, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})
    boundary_distance_threshold: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': True, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})

    # second set of parameters have advanced True and training False and will be shown in the advanced popup dialog
    # None yet..
    
    # third set of parameters have advanced False and training True and will be shown in the training popup dialog
    num_epochs: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
    model_name: str = field(metadata={'type': 'str', 'harvest': True, 'advanced': False, 'training': True, 'default': 'cyto3', 'step': 1})
        
    
    descriptor = "Micro-sam Instance Framework"

    def __init__(self, parent_path: str,  num_classes: int, start_model: str = None):
        super().__init__(parent_path, num_classes)
        
        self.model = None 

        # microsam models are stored in a directory        
        self.load_mode = LoadMode.Directory
        
        self.num_epochs = 100
    
        # initial model names
        self.model_names = ['vit_b_lm', 'vit_b', 'vit_b_em_organelles']
        
        # pretrained model names
        self.builtin_names = ['vit_b', 'vit_b_lm', 'vit_b_em_organelles']
        
        self.model_type = "vit_b_lm"
        self.model_name = "vit_b_lm" #self.generate_model_name('microsam')
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # the device/GPU used for training

        self.tile_size = 384  # the size of the tiles used for training
        self.halo_size = 64  # the size of the halo used for training
        self.prediction_channel = 1  # the channel used for prediction

        self.foreground_threshold = 0.5  # the threshold for foreground pixels
        self.center_distance_threshold = 0.5  # the threshold for center distance
        self.boundary_distance_threshold = 0.5  # the threshold for boundary distance

        
    def train(self, updater=None):
        """
        Train the Micro-sam model

        The training patches should already exist in the patch_path directory.
        """

        if updater is None:
            updater = self.updater
        
        updater('Training Microsam model', 0)

        image_dir = os.path.join(self.patch_path, 'input0')
        image_dir_255 = tempfile.mkdtemp(dir=self.patch_path)

        image_paths = sorted(glob.glob(os.path.join(image_dir, "*")))

        for image_path in image_paths:
            image = imageio.imread(image_path)
            image = (image * 255).astype('uint8')
            imageio.imwrite(image_path.replace(image_dir, image_dir_255), image)

        patch_shape = [1, image.shape[0], image.shape[1]]  # Assuming the images are 2D, we set the third dimension to 1
        
        try:
            segmentation_dir = os.path.join(self.patch_path, 'ground truth0')

            # Load images from multiple files in folder via pattern (here: all tif files)
            raw_key, label_key = "*.tif", "*.tif"

            # The 'roi' argument can be used to subselect parts of the data.
            # Here, we use it to select the first 70 images (frames) for the train split and the other frames for the val split.
            # Todo: figure out why roi does not work with RGB images
            train_roi = None #np.s_[:, :, :]
            val_roi = None #np.s_[:5, :, :]

            batch_size = 1  # the training batch size

            train_instance_segmentation = True

            sampler = MinInstanceSampler(min_size=25)  # NOTE: The choice of 'min_size' value is paired with the same value in 'min_size' filter in 'label_transform'.

            train_loader = sam_training.default_sam_loader(
                raw_paths=image_dir_255,
                raw_key=raw_key,
                label_paths=segmentation_dir,
                label_key=label_key,
                with_segmentation_decoder=train_instance_segmentation,
                patch_shape=patch_shape,
                batch_size=batch_size,
                is_seg_dataset=True,
                rois=train_roi,
                shuffle=True,
                raw_transform=sam_training.identity,
                sampler=sampler,
            )

            val_loader = sam_training.default_sam_loader(
                raw_paths=image_dir_255,
                raw_key=raw_key,
                label_paths=segmentation_dir,
                label_key=label_key,
                with_segmentation_decoder=train_instance_segmentation,
                patch_shape=patch_shape,
                batch_size=batch_size,
                is_seg_dataset=True,
                rois=val_roi,
                shuffle=True,
                raw_transform=sam_training.identity,
                sampler=sampler,
            )
            
            # All hyperparameters for training.
            n_objects_per_batch = 2  # the number of objects per batch that will be sampled

            # The model_type determines which base model is used to initialize the weights that are finetuned.
            # We use vit_b here because it can be trained faster. Note that vit_h usually yields higher quality results.
            model_type = self.model_type  # 'vit_b_lm', 'vit_b', 'vit_b_em_organelles'
            print(f"Training with device: {self.device}, model_type: {model_type}, n_objects_per_batch: {n_objects_per_batch}")
            # Run training (best metric 0.027211

            if updater is not None:
                signals = ProgressSignals()
                custom_callback = CustomCallback(updater, num_epochs=self.num_epochs)
                signals.pbar_total.connect(custom_callback.set_total)
                signals.pbar_update.connect(custom_callback.on_epoch_update)
            else:
                signals = None

            sam_training.train_sam(
                name=self.model_name,
                save_root=self.model_path, #os.path.join(root_dir, "models"),
                model_type=model_type,
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=self.num_epochs,
                n_objects_per_batch=n_objects_per_batch,
                with_segmentation_decoder=train_instance_segmentation,
                device=self.device,
                #n_iterations=10,
                pbar_signals=signals
            )
        finally:
            # Clean up the temporary directory
            shutil.rmtree(image_dir_255)



    def predict(self, img: np.ndarray):
        """
        Predict the segmentation of the image using the current microsam model
        """
        print("Predicting using Micro-sam model")
        if self.model_name not in self.builtin_names:
            best_checkpoint =  os.path.join(self.model_path, 'checkpoints', self.model_name, 'best.pt')
        else:
            best_checkpoint = None

        #img_ = np.transpose(img, (2, 0, 1))  # Convert image to (C, H, W) format if needed

        prediction = self.run_automatic_instance_segmentation(
            image=img, checkpoint_path=best_checkpoint, model_type=self.model_type, device=self.device, tile_shape=(self.tile_size, self.tile_size), halo = (self.halo_size, self.halo_size)
        )

        return prediction 
    
    def run_automatic_instance_segmentation(self, image, checkpoint_path, model_type="vit_b_lm", device=None, tile_shape = None, halo = None):
        """Automatic Instance Segmentation (AIS) by training an additional instance decoder in SAM.

        NOTE: AIS is supported only for `µsam` models.

        Args:
            image: The input image.
            checkpoint_path: The path to stored checkpoints.
            model_type: The choice of the `µsam` model.
            device: The device to run the model inference.

        Returns:
            The instance segmentation.
        """
        # Step 1: Get the 'predictor' and 'segmenter' to perform automatic instance segmentation.
        predictor, segmenter = get_predictor_and_segmenter(
            model_type=model_type, # choice of the Segment Anything model
            checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
            device=device,  # the device to run the model inference.
            is_tiled = (tile_shape is not None),  # whether the model is tiled or not.
        )

        # Step 2: Get the instance segmentation for the given image.
        prediction = automatic_instance_segmentation(
            predictor=predictor,  # the predictor for the Segment Anything model.
            segmenter=segmenter,  # the segmenter class responsible for generating predictions.
            input_path=image,
            ndim=2,
            tile_shape=tile_shape,
            halo=halo,
            foreground_threshold=self.foreground_threshold,  # the threshold for foreground pixels.
            center_distance_threshold=self.center_distance_threshold,  # the threshold for center distance.
            boundary_distance_threshold=self.boundary_distance_threshold,  # the threshold for boundary distance.
        )

        return prediction
    
    
    def get_model_names(self):
        return self.model_names 
    
    def set_builtin_model(self, model_name):
        self.model_type = model_name
        self.model_name = model_name 
    
    def load_model_from_disk(self, model_name):

        self.model_name = model_name
        
    def create_callback(self, updater):
        self.updater = updater
    


# this line is needed to register the framework on import
BaseFramework.register_framework('MicroSamInstanceFramework', MicroSamInstanceFramework)
