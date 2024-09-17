
class BaseModel:
    def __init__(self, patch_path, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.patch_path = patch_path

    def set_model(self, model):
        self.model = model

    def train(self, num_epochs, updater=None):
        pass

    def predict(self, image):
        pass