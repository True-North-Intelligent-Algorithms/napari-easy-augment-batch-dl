from datetime import datetime
import os 
from dataclasses import dataclass

class LoadMode:
    NotLoadable = 0
    Directory = 1
    File=2

class TrainMode:
    Patches = 0
    Pixels = 1

@dataclass
class BaseFramework:
    """
    Base class for frameworks. 

    Frameworks represents DL approaches like Stardist, Cellpose, SAM or Semantic Segmentation Unets.

    The frameworks are intended to work with a specific pre-existing file organization

    parent_path: path to the parent directory, the images are stored in parent_path/images
    parent_path/models: path to the models directory, where the models are stored
    parent_path/patches: path to the patches directory, where the patches are stored
    """

    # initiate registry for all the frameworks. Each framework should register itself when imported:
    # For example a cellpose framework would have the following line in its code:
    # BaseFramework.register_framework('CellPoseInstanceFramework', CellPoseInstanceFramework)
    registry = {}

    @classmethod
    def register_framework(cls, name, framework):
        """
        Add this a framework to the registry. 

        Args:
            name (str): The name of the framework.
            framework (BaseFramework): The framework class to register.
        """
        cls.registry[name] = framework

    def __init__(self, parent_path, num_classes=1):
        """
        Base class for all frameworks.
        """
        
        self.parent_path = parent_path
        # path to store models
        self.model_path = os.path.join(parent_path, 'models') 
        self.num_classes = num_classes
        # path to store patches
        self.patch_path = os.path.join(parent_path, 'patches')
        
        # current model name
        self.model_name = 'notset'
        self.load_mode = LoadMode.NotLoadable
        
        # boxes should be set true if the framework detects bounding boxes
        self.boxes = False

        # builtins are models that are included with the framework (for example cyto3 in cellpose)
        self.builtin_names = []
        
        # loss tracking lists
        self.train_loss_list = []
        self.validation_loss_list = []

        # model dictionary stores all models including builtins and custom models
        self.model_dictionary = {}
        self.train_mode = TrainMode.Patches

    def get_image_label_files(self, patch_path, input_str, ground_truth_str, num_truths):
        """
        Collect image and label files from patch directory.
        
        Args:
            patch_path (Path): Path object pointing to the patch directory containing image and label subdirectories.
            input_str (str): Name of the input image subdirectory (e.g., 'input0', 'input_validation0').
            ground_truth_str (str): Base name of the ground truth label subdirectories (e.g., 'ground truth').
                                   Numbers will be appended (ground truth0, ground truth1, etc.).
            num_truths (int): Number of ground truth label classes to collect.
        
        Returns:
            tuple: A tuple (X, Y) where:
                - X (list): Sorted list of Path objects for all .tif files in the input directory.
                - Y (list of lists): List where each element is a sorted list of Path objects for .tif files
                                    in each ground truth directory (ground truth0, ground truth1, etc.).
        """
        X = sorted(patch_path.rglob(f'**/{input_str}/*.tif'))
        Y = []
        for i in range(num_truths):
            Y.append(sorted(patch_path.rglob(f'**/{ground_truth_str}{i}/*.tif')))
        return X, Y

    def train(self, updater=None):
        """
        Train the model.

        This method must be implemented by derived classes. It defines the
        process for training the model, potentially using an optional updater
        for progress reporting.

        Args:
            updater (callable, optional): A callback function that can be
                called during training to report progress.
        """
        pass

    def predict(self, image):
        """
        Predict the output for a given image.

        This method must be implemented by derived classes. It defines the
        process for making predictions on input data.

        Args:
            image: The input image as a numpy array.

        Returns:
            The prediction result. 
        """
        pass

    def get_model_names(self):
        """
        Get the names of all models that are available for this framework.
        
        Override this method in derived classes to return a list of model names.

        Often the list will include builtins and custom models.
        """
        return ['notset']

    def get_optimizers(self):
        """
        Get names of available optimizers.

        (optional) Override this method in derived classes to return a list of optimizer names.
        """
        return []

    def create_callback(self, updater):
        """
        Create a callback function for training.  The updater function may be wrapped in a custom callback class

        For example in a stardist trainer a keras.callbacks.Callback is created that calls the updater function.

        In other frameworks the updater function may be used directly by the training function.

        Args:
            updater (callable): A callback function with parameters (message, progress)
        """
        self.updater = updater

    def generate_model_name(self, base_name="model"):
        """
        Generate a unique model name based on the current time.

        May be overridden in derived classes to provide a custom naming scheme.

        Args:
            base_name (str): The base name for the model.

        Returns:
            str: The generated model name.
        """
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}"
        return model_name
    
    def set_pretrained_model(self, model_name):
        """
        Sets a pretrained model.  Built in models are models which are included with the framework and are created by the framework developers.

        If the model name is a built_in model, the model is created by calling the set_builtin_model method.

        For example in Cellpose a Cyto3 model is a built in model. And can be create by calling 'models.CellposeModel(gpu=True, model_type='cyto3')'

        For cellpose frame 'set_builtin_model' will be overridden with the above code.

        Args:
            model_name (str): The name of the model to set
        """
        if model_name != 'notset':

            model = self.model_dictionary.get(model_name, None)

            if model is None:

                # if a built in model set it
                if model_name in self.builtin_names:
                    self.set_builtin_model(model_name)
                    self.model_dictionary[model_name] = self.model
            else:
                self.model = model

    def set_optimizer(self, optimizer):
        """
        Set the optimizer for the model.
        """
        pass

    def set_builtin_model(self, model_name):
        """
        Built in models are models which are included with the framework and are created by the framework developers.

        For example in Cellpose a Cyto3 model is a built in model. And can be create by calling 'models.CellposeModel(gpu=True, model_type='cyto3')'
        
        For cellpose frame 'set_builtin_model' will be overridden with the above code.

        Each framework can optionally override this method to set a built in model using the protocol of the framework.
        Args:
            model_name (str): The name of the model to set
        """
        pass

