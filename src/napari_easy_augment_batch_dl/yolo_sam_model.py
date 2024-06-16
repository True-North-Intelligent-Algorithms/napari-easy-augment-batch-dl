from napari_easy_augment_batch_dl.base_model import BaseModel
import numpy as np
import os
from segment_everything.prompt_generator import YoloDetector
from segment_everything.weights_helper import get_weights_path
from segment_everything.stacked_labels import StackedLabels
from segment_everything.detect_and_segment import segment_from_stacked_labels

class YoloSAMModel(BaseModel):
    def __init__(self, patch_path: str, model_path: str,  num_classes: int, start_model_path: str = None):
        super().__init__(patch_path, model_path, num_classes)
        #self.yolo_detecter = YoloDetector( 'yolov8m.pt', "RegularYOLO", 'cuda')
        self.yolo_detecter = YoloDetector(str(get_weights_path("ObjectAwareModel")), "ObjectAwareModelFromMobileSamV2", device='cuda')
        pass
                 
    def predict(self, img: np.ndarray):
        results = self.yolo_detecter.get_results(img, conf=0.1, iou=0.8, imgsz=1024)
        self.bbs=results[0].boxes.xyxy.cpu().numpy()
        stacked_labels = StackedLabels.from_yolo_results(self.bbs, None, img)
        segmented_stacked_labels = segment_from_stacked_labels(stacked_labels, "MobileSamV2")
        labels = segmented_stacked_labels.make_2d_labels(type="min")
        return labels, self.bbs
    
    def train(self, num_epochs, updater=None):
        pass