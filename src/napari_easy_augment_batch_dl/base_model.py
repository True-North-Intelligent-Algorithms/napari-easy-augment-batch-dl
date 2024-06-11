
class BaseModel:
    def __init__(self, patch_path, model_path, num_classes):
        self.model_path = model_path
        self.num_classes = num_classes
        self.patch_path = patch_path

    def train(self, updater=None):
        pass

    def predict(self, image):
        pass