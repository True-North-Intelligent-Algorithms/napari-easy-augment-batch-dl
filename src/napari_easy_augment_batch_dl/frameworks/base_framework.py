from datetime import datetime
import os 

class LoadMode:
    NotLoadable = 0
    Directory = 1
    File=2

class BaseFramework:
    def __init__(self, parent_path, num_classes=1):
        self.model_path = os.path.join(parent_path, 'models') 
        self.num_classes = num_classes
        self.patch_path = os.path.join(parent_path, 'patches')
        self.model_name = 'notset'
        self.load_mode = LoadMode.NotLoadable
        self.boxes = False
        self.builtin_names = []
        self.pretrained_models = {}

    def set_model(self, model):
        self.model = model

    def train(self, num_epochs, updater=None):
        pass

    def predict(self, image):
        pass

    def get_model_names(self):
        return ['notset']

    def get_optimizers(self):
        return []

    def create_callback(self, updater):
        self.updater = updater

    def generate_model_name(self, base_name="model"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}"
        return model_name
    
    def set_pretrained_model(self, model_name):
        if model_name != 'notset':

            model = self.pretrained_models.get(model_name, None)

            if model is None:

                # if a built in model set it
                if model_name in self.builtin_names:
                    #self.model = models.CellposeModel(gpu=True, model_type=model_name)
                    self.set_builtin_model(model_name)
                    self.pretrained_models[model_name] = self.model
            else:
                self.model = model

    def set_optimizer(self, optimizer):
        pass

    def set_builtin_model(self, model_name):
        pass

