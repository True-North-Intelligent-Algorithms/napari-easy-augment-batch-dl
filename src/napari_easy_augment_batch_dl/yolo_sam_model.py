from napari_easy_augment_batch_dl.base_model import BaseModel
import numpy as np
import os
from segment_everything.prompt_generator import YoloDetector
from segment_everything.weights_helper import get_weights_path
from segment_everything.stacked_labels import StackedLabels
from segment_everything.detect_and_segment import segment_from_stacked_labels
import sys

class YoloSAMModel(BaseModel):
    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model_path: str = None):
        super().__init__(patch_path, model_path, num_classes)
        #self.yolo_detecter = YoloDetector( 'yolov8m.pt', "RegularYOLO", 'cuda')
        model_name = os.path.join(model_path, r'')
        
        if start_model_path is not None:
            best_model_path = os.path.join(start_model_path, 'weights', 'best.pt')
            self.yolo_detecter = YoloDetector(best_model_path, "loaded YOLO", device='cuda')
        self.custom_model = None
                 
    def predict(self, img: np.ndarray, imagesz=1024):
        results = self.yolo_detecter.get_results(img, conf=0.1, iou=0.8, imgsz=imagesz)
        self.bbs=results[0].boxes.xyxy.cpu().numpy()
        stacked_labels = StackedLabels.from_yolo_results(self.bbs, None, img)
        segmented_stacked_labels = segment_from_stacked_labels(stacked_labels, "MobileSamV2")
        labels = segmented_stacked_labels.make_2d_labels(type="min")
        return labels, self.bbs
    
    def create_callback(self, updater):
        self.updater = updater
        pass
    
    def train(self, num_epochs, updater=None):
        #if 'ultralytics' in sys.modules:
        #    del sys.modules['ultralytics']
        from ultralytics import YOLO
        yaml_name = os.path.join(os.path.dirname(self.patch_path), "data.yaml")
        print(yaml_name)
        
        if self.custom_model is None:
            self.custom_model = YOLO('yolov8m')
            resume = False
        else:
            resume = True

        project_path = os.path.join(self.model_path, 'YOLO-training')
        name = '100-epoch-model'

        # Train the model
        results = self.custom_model.train(data=yaml_name,
            project=project_path,
            name=name,
            epochs=num_epochs,
            patience=0, #I am setting patience=0 to disable early stopping.
            batch=50,
            workers=1,
            imgsz=1024,
            scale = 0.9,
            degrees = 90,
            shear = 90,
            hsv_h = 0.4,
            resume = resume)
        
        new_model_path = os.path.join(project_path, name, 'weights', 'best.pt')
        self.yolo_detecter = YoloDetector(new_model_path, "new trained YOLO", device='cuda')
