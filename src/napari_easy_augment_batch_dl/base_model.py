from datetime import datetime

class DLModel:
    UNET = "U-Net"
    STARDIST = "Stardist"
    CELLPOSE = "CellPose"
    YOLO_SAM = "Yolo/SAM"
    MOBILE_SAM2 = "Mobile SAM 2"

class LoadMode:
    NotLoadable = 0
    Directory = 1
    File=2


class BaseModel:
    def __init__(self, patch_path='', model_path='', num_classes=1):
        self.model_path = model_path
        self.num_classes = num_classes
        self.patch_path = patch_path
        self.model_name = 'notset'
        self.load_mode = LoadMode.NotLoadable
        self.boxes = False

    def set_model(self, model):
        self.model = model

    def train(self, num_epochs, updater=None):
        pass

    def predict(self, image):
        pass

    def get_model_names(self):
        return ['notset']
    
    def create_callback(self, updater):
        pass

    def generate_model_name(self, base_name="model"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}"
        return model_name
    
    def set_pretrained_model(self, model_path):
        pass