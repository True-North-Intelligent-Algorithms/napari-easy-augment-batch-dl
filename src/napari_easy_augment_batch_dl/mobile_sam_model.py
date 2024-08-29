from napari_easy_augment_batch_dl.base_model import BaseModel
import numpy as np
import os
from segment_everything.prompt_generator import YoloDetector
from segment_everything.weights_helper import get_weights_path
from segment_everything.stacked_labels import StackedLabels
from segment_everything.detect_and_segment import segment_from_stacked_labels
import sys

class MobileSAMModel(BaseModel):
    def __init__(self, patch_path: str, model_path: str, start_model_path: str = None):
        super().__init__(patch_path, model_path, 1)
        self.yolo_detecter = YoloDetector(str(get_weights_path("ObjectAwareModel")), "ObjectAwareModelFromMobileSamV2", device='cuda')
                 
    def predict(self, img: np.ndarray, imagesz=1024):
        results = self.yolo_detecter.get_results(img, conf=0.1, iou=0.8, imgsz=imagesz, max_det=10000)
        self.bbs=results[0].boxes.xyxy.cpu().numpy()
        stacked_labels = StackedLabels.from_yolo_results(self.bbs, None, img)
        segmented_stacked_labels = segment_from_stacked_labels(stacked_labels, "MobileSamV2")
        labels = segmented_stacked_labels.make_2d_labels(type="min")
        return labels, self.bbs
    
    def train(self, num_epochs, updater=None):
        # raise not implemented error
        raise NotImplementedError("This model is not trainable")
         