from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework
import numpy as np
from segment_everything.prompt_generator import YoloDetector
from segment_everything.weights_helper import get_weights_path
from segment_everything.stacked_labels import StackedLabels
from segment_everything.detect_and_segment import segment_from_stacked_labels
from dataclasses import dataclass, field

@dataclass
class MobileSAMFramework(BaseFramework):

    conf: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.5, 'step': 0.1})
    iou: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': 0.0, 'max': 1.0, 'default': 0.8, 'step': 0.1})
    imagesz: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 10000, 'default': 1024, 'step': 1})
    
    num_epochs: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
        
    descriptor = "MobileSAM Model"

    def __init__(self, parent_path, start_model_path: str = None):
        super().__init__(parent_path, 1)
        self.yolo_detecter = YoloDetector(str(get_weights_path("ObjectAwareModel")), "ObjectAwareModelFromMobileSamV2", device='cuda')
        
        self.conf = 0.5
        self.iou = 0.8
        self.imagesz = 1024
        self.boxes = True
                 
    def predict(self, img: np.ndarray):
        results = self.yolo_detecter.get_results(img, conf=self.conf, iou= self.iou, imgsz=self.imagesz, max_det=10000)
        self.bbs=results[0].boxes.xyxy.cpu().numpy()
        stacked_labels = StackedLabels.from_yolo_results(self.bbs, None, img)
        segmented_stacked_labels = segment_from_stacked_labels(stacked_labels, "MobileSamV2")
        segmented_stacked_labels.sort_largest_to_smallest()
        labels = segmented_stacked_labels.make_2d_labels(type="min")
        return labels, self.bbs
    
    def train(self, updater=None):
        # raise not implemented error
        raise NotImplementedError("This model is not trainable")

BaseFramework.register_framework('MobileSAMFramework', MobileSAMFramework)
         