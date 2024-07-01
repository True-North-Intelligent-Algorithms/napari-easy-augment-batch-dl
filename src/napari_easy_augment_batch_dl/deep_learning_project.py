from pathlib import Path
import os
from skimage.io import imread, imsave
import numpy as np
from tnia.deeplearning.dl_helper import generate_patch_names, generate_next_patch_name
import json
from tnia.deeplearning.dl_helper import make_label_directory
from tnia.deeplearning.augmentation import uber_augmenter, uber_augmenter_bb
from tnia.deeplearning.dl_helper import quantile_normalization
from napari_easy_augment_batch_dl.bounding_box_util import is_bbox_within, xyxy_to_normalized_xywh
import pandas as pd
import yaml
import glob 


try:
    from napari_easy_augment_batch_dl.pytorch_semantic_model import PytorchSemanticModel
except ImportError:
    PytorchSemanticModel = None
try:
    from napari_easy_augment_batch_dl.stardist_instance_model import StardistInstanceModel
except ImportError:
    StardistInstanceModel = None
try:
    from napari_easy_augment_batch_dl.cellpose_instance_model import CellPoseInstanceModel
except ImportError:
    CellPoseInstanceModel = None
try:
    from napari_easy_augment_batch_dl.mobile_sam_model import MobileSAMModel
except ImportError:
    MobileSAMModel = None
try:
    from napari_easy_augment_batch_dl.yolo_sam_model import YoloSAMModel
except ImportError:
    YoloSAMModel = None

class DLModel:
    UNET = "U-Net"
    STARDIST = "Stardist"
    CELLPOSE = "CellPose"
    YOLO_SAM = "Yolo/SAM"
    MOBILE_SAM2 = "Mobile SAM 2"

class DeepLearningProject:
    def __init__(self, parent_path, num_classes=1):

        self.models = {}
        self.models[DLModel.UNET] = None
        self.models[DLModel.STARDIST] = None
        self.models[DLModel.CELLPOSE] = None
        self.models[DLModel.YOLO_SAM] = None
        self.models[DLModel.MOBILE_SAM2] = None

        self.parent_path = Path(parent_path)

        # check if json.info exists
        json_name = parent_path / 'info.json'
        if os.path.exists(json_name):
            with open(json_name, 'r') as f:
                json_ = json.load(f)
                self.num_classes = json_['num_classes']
        else:    
            self.num_classes = num_classes

        self.image_folder = Path(parent_path)

        self.image_path = Path(parent_path)
        self.label_path = Path(parent_path / r'labels')
        self.napari_path = Path(parent_path / r'napari')

        self.patch_path= self.parent_path / 'patches'
        self.model_path = self.parent_path / 'models'

        self.yolo_label_path = Path(parent_path / r'yolo_labels')
        self.yolo_patch_path = Path(parent_path / r'yolo_patches')

        if not os.path.exists(self.patch_path):
            os.mkdir(self.patch_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.yolo_label_path):
            os.mkdir(self.yolo_label_path)
        if not os.path.exists(self.yolo_patch_path):
            os.mkdir(self.yolo_patch_path)
        
        self.files = list(self.image_path.glob('*.jpg'))
        self.files = self.files+list(self.image_path.glob('*.jpeg'))
        self.files = self.files+list(self.image_path.glob('*.tif'))
        self.files = self.files+list(self.image_path.glob('*.png'))
        
        self.image_label_paths, self.mask_label_paths = make_label_directory(1, self.num_classes, self.label_path)
    
        self.image_list = []
        for index in range(len(self.files)):
            im = imread(self.files[index])
            print(im.shape)
            if len(im.shape) == 3:
                if im.shape[2] > 3:
                    im = im[:,:,:3]
            self.image_list.append(im)
        
        self.label_list = []
        self.prediction_list = []
        
        self.boxes = []
        self.object_boxes = None
        self.features = None
        
        if os.path.exists(json_name):

            for c in range(self.num_classes):
                labels_temp = []
                predictions_temp = []
                
                label_names = list(Path(self.mask_label_paths[c]).glob('*.tif'))
                json_names = list(Path(self.image_label_paths[c]).glob('*.json'))
                print('there are {} labels for class {}'.format(len(label_names), c))
                print('there are {} json files for class {}'.format(len(json_names), c))

                n=0            
                for image_name, image in zip(self.files, self.image_list):
                    image_base_name = image_name.name.split('.')[0]

                    # get all label names for this image
                    label_names_ = [x for x in label_names if image_base_name in x.name]

                    # get all json names for this image
                    json_names_ = [x for x in json_names if image_base_name in x.name]

                    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                    prediction = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

                    print('image base name is ', image_base_name)

                    for label_name_, json_name_ in zip(label_names_, json_names_):
                        with open(json_name_, 'r') as f:
                            json_ = json.load(f)
                            print(json_)
                            
                            x1= json_['bbox'][0]
                            y1= json_['bbox'][1]
                            x2= json_['bbox'][2]
                            y2= json_['bbox'][3]

                            tl = [n, y1, x1]
                            tr = [n, y1, x2]
                            br = [n, y2, x2]
                            bl = [n, y2, x1]
                            bbox = [tl, tr, br, bl]
                            self.boxes.append(bbox)

                            label_crop = imread(label_name_)
                            label[y1:y2, x1:x2] = label_crop
                            #rois.append([[x1, y1], [x2, y2]])

                    labels_temp.append(label)
                    predictions_temp.append(prediction)
                    n = n+1

            self.label_list.append(labels_temp)
            self.prediction_list.append(predictions_temp)                        

        else:
            for c in range(self.num_classes):
                self.label_list.append([])
                self.prediction_list.append([])
                for image in self.image_list:
                    self.label_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))
                    self.prediction_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))
        
        '''
        #self.image_files = self.load_image_files()
        if self.napari_path.exists():
            self.images = np.load(self.napari_path / 'images.npy')

            self.label_list = []
            self.prediction_list = []
            
            for c in range(self.num_classes):
                self.label_list.append(np.load(os.path.join(self.napari_path, 'labels_'+str(c)+'.npy')))
                self.prediction_list.append(np.load(os.path.join(self.napari_path, 'predictions_'+str(c)+'.npy')))
            try:
                self.boxes = np.load(self.napari_path / 'Label box.npy')
            except:
                self.boxes = []

            try:
                self.object_boxes = np.load(self.napari_path / 'Object box.npy')
            except:
                self.object_boxes = []

            try:
                self.features = pd.read_csv(self.napari_path / 'features.csv') 
            except:
                self.features = []           
        else:
            self.initialize_napari_project()
        '''        

    def initialize_napari_project(self):
        print('Napari directory does not exist, initializing project')

        self.label_list = []
        self.prediction_list = []

        for c in range(self.num_classes):
            self.label_list.append([])
            self.prediction_list.append([])

        for im in self.image_list:

            for c in range(self.num_classes): 
                self.label_list[c].append(np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8))
                self.prediction_list[c].append(np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8))

        self.images = self.pad_to_largest(self.image_list) #np.array(self.image_list)
        #self.images = np.array(self.image_list)

        for c in range(self.num_classes):
            self.label_list[c] =  self.pad_to_largest(self.label_list[c])#np.array(self.label_list[c])
            self.prediction_list[c] = self.pad_to_largest(self.prediction_list[c])
            #self.label_list[c] = np.array(self.label_list[c])

        self.boxes = None
        self.object_boxes = None
        self.features = None
    
    def delete_all_files_in_directory(self, directory_path):
        # Get a list of all files in the directory
        files = glob.glob(os.path.join(directory_path, '*'))
        
        # Iterate over the list of files and remove each one
        for file in files:
            if os.path.isfile(file):  # Check if it is a file
                os.remove(file)
                print(f'Deleted file: {file}')
            else:
                print(f'Skipped non-file item: {file}')

    def save_project(self, boxes):

        # save json file with num_classes
        json_name = os.path.join(self.parent_path, 'info.json')

        with open(json_name, 'w') as f:
            json_ = {}
            json_['num_classes'] = self.num_classes
            json.dump(json_, f)

        for c in range(self.num_classes):
            self.delete_all_files_in_directory(self.image_label_paths[c])
            self.delete_all_files_in_directory(self.mask_label_paths[c])
        
        for box in boxes:
            z = int(box[0,0])

            name = self.files[z].name.split('.')[0]

            # get rid of first column (z axis)
            box = box[:,1:]

            ystart = int(np.min(box[:,0]))
            yend = int(np.max(box[:,0]))
            xstart = int(np.min(box[:,1]))
            xend = int(np.max(box[:,1]))

            #print('bounding box is',ystart, yend, xstart, xend)
            print('image file is ', self.files[z])

            if np.ndim(self.image_list[z]) == 3:
                im = self.image_list[z][ystart:yend, xstart:xend, :]
            else:
                im = self.image_list[z][ystart:yend, xstart:xend]

            labels=[]
            
            for c in range(self.num_classes):
                labels.append(self.label_list[c][z][ystart:yend, xstart:xend])
                print('labelsum is ', labels[c].sum())

            print(im.shape, labels[0].shape)

            image_name, mask_name = generate_patch_names(str(self.image_label_paths[0]), str(self.mask_label_paths[0]), name)
            base_name = generate_next_patch_name(str(self.image_label_paths[0]), name)

            print(base_name)
            print(image_name)
            print(mask_name)

            imsave(image_name, im)
            print(image_name)

            for c in range(self.num_classes):
              print(self.mask_label_paths[c])
              imsave(os.path.join(self.mask_label_paths[c], base_name+".tif"), labels[c])

            # save xstart, ystart, xend, yend to json 
            json_name = os.path.join(self.image_label_paths[0], base_name+".json")
            with open(json_name, 'w') as f:
                json_ = {}
                json_['base_name'] = base_name
                json_['bbox'] = [xstart, ystart, xend, yend]
                json.dump(json_, f)
    
    def delete_augmentations(self):
        image_patch_path =  os.path.join(self.patch_path, 'input0')
        self.delete_all_files_in_directory(image_patch_path)

        for c in range(self.num_classes):
            label_patch_path =  os.path.join(self.patch_path, 'ground truth'+str(c))
            self.delete_all_files_in_directory(label_patch_path) 

    def perform_augmentation(self, boxes, num_patches = 100, patch_size=256, updater = None, 
                                  do_horizontal_flip=True, do_vertical_flip=True, do_random_rotate90=True, do_random_sized_crop=True, 
                                  do_random_brightness_contrast=True, do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):

        patch_path= self.parent_path / 'patches' 

        if not os.path.exists(patch_path):
            os.mkdir(patch_path)

        for box in boxes:
            z = int(box[0,0])

            name = self.files[z].name.split('.')[0]

            # get rid of first column (z axis)
            box = box[:,1:]

            ystart = int(np.min(box[:,0]))
            yend = int(np.max(box[:,0]))
            xstart = int(np.min(box[:,1]))
            xend = int(np.max(box[:,1]))

            if np.ndim(self.image_list[z]) == 3:
                im = self.image_list[z][ystart:yend, xstart:xend, :]
            else:
                im = self.image_list[z][ystart:yend, xstart:xend]

            labels=[]
            
            for c in range(self.num_classes):
                labels.append(self.label_list[c][z][ystart:yend, xstart:xend])

            ## IMPORTANT:  Here is where normalization 1 is done when applying network need to match. 
            im = quantile_normalization(im, quantile_low=0.001).astype(np.float32)

            #do_random_sized_crop=True, do_random_brightness_contrast=True, 
            #do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False)
            uber_augmenter(im, labels, patch_path, 'grid', patch_size, num_patches, do_horizontal_flip=do_horizontal_flip, do_vertical_flip=do_vertical_flip, do_random_rotate90=do_random_rotate90, do_random_sized_crop=do_random_sized_crop, do_random_brightness_contrast=do_random_brightness_contrast, 
                                  do_random_gamma=do_random_gamma, do_color_jitter=do_color_jitter, do_elastic_transform=do_elastic_transform)
       
    def perform_yolo_augmentation(self, boxes, objects, classes, num_patches_per_image, patch_size, updater=None,
                                  do_horizontal_flip=True, do_vertical_flip=True, do_random_rotate90=True, do_random_sized_crop=True, 
                                  do_random_brightness_contrast=True, do_random_gamma=False, do_color_jitter=False):
        """ Yolo Bounding box augmentation is a little different than the pixel mask augmentation

            Instead of cropping ROIs and corresponding masks, we mask the pixels outside the rois to the mean of the pixels inside the ROIs
            This is done to preserve the scale of the objects with respect to the image size.  We then augment this masked image        

            For Yolo we don't need a consistent patch size, because the image will be resized during training. 
        
        Args:
            boxes (_type_): bounding boxes for the rois
            objects (_type_): bounding boxes for the objects
            num_patches_per_roi (_type_): number of patches to make per image
            updater (_type_, optional):  Update the GUI with status messages and progress. Defaults to None.
        """




        # unlike the pixel mask DL we don't need to create a separate ground truth folder for each class
        # so when we make the label directores num_classes is 1
        #self.yolo_image_label_paths, self.yolo_mask_label_paths = make_label_directory(1, 1, self.yolo_label_path)
        #self.yolo_image_patch_paths, self.yolo_mask_patch_paths = make_label_directory(1, 1, self.yolo_patch_path)
        self.yolo_image_label_paths = [os.path.join(self.yolo_label_path, 'images')]
        self.yolo_mask_label_paths = [os.path.join(self.yolo_label_path, 'labels')]
        self.yolo_image_patch_paths = [os.path.join(self.yolo_patch_path, 'images')]
        self.yolo_mask_patch_paths = [os.path.join(self.yolo_patch_path, 'labels')]

        if not os.path.exists(self.yolo_image_label_paths[0]):
            os.mkdir(self.yolo_image_label_paths[0])
        if not os.path.exists(self.yolo_mask_label_paths[0]):
            os.mkdir(self.yolo_mask_label_paths[0])
        if not os.path.exists(self.yolo_image_patch_paths[0]):
            os.mkdir(self.yolo_image_patch_paths[0])
        if not os.path.exists(self.yolo_mask_patch_paths[0]):
            os.mkdir(self.yolo_mask_patch_paths[0])
       
        # write the yaml 
        # Create the YAML structure
        names = ['c'+str(i) for i in range(self.num_classes)]
        
        data = {
            'names': names,
            'nc': self.num_classes,
            'train': self.yolo_image_patch_paths[0],
            'val': self.yolo_image_patch_paths[0]
        }

        # Write the YAML file
        with open(os.path.join(self.parent_path, 'data.yaml'), 'w') as f:
            yaml.dump(data, f)

        updater('Performing Yolo Augmentation', 0)

        boxes = np.array(boxes)
        objects = np.array(objects)

        num_images = len(self.image_list)

        for n in range(num_images):
            
            index_objects = np.all(objects[:,:,0]==n, axis=1)
            filtered_objects = objects[index_objects]
            filtered_classes = classes[index_objects]

            if len(boxes) == 0:
                filtered_boxes = []
            else:
                index_boxes = np.all(boxes[:,:,0]==n, axis=1)
                filtered_boxes = boxes[index_boxes]

            print('number of boxes at',n,' is ', len(filtered_boxes))
            print('number of objects at',n,' is ', len(filtered_objects))

            # save the image to yolo label path
            im = self.image_list[n]

            image_height = im.shape[0]
            image_width = im.shape[1]

            if len(filtered_objects > 0):
                
                if len(filtered_boxes) > 0:
                    print(len(filtered_boxes), ' boxes for image ', n)
                    mask = np.zeros_like(im)
                    # use bounding box to create mask
                    for box in filtered_boxes:
                        ystart = int(np.min(box[:,1]))
                        yend = int(np.max(box[:,1]))
                        xstart = int(np.min(box[:,2]))
                        xend = int(np.max(box[:,2]))

                        mask[ystart:yend, xstart:xend] = 1

                    im = np.where(mask>0, im, np.mean(im)).astype(im.dtype)
                
                imsave(os.path.join(self.yolo_image_label_paths[0], self.files[n].name), im)

                # create text file for image n
                base_name = self.files[n].name.split('.')[0]
                yolo_name = os.path.join(self.yolo_mask_label_paths[0],(base_name + '.txt'))
                boxes_xywhn = []
                with open(yolo_name, 'w') as f:
                    for object, class_ in zip(filtered_objects, filtered_classes):
                        xywhn = xyxy_to_normalized_xywh(object[:,1:], image_width, image_height)
                        f.write(str(class_)+' ')
                        xywhn_str = str(xywhn).replace('(','').replace(')','').replace(',','')
                        f.write(xywhn_str)
                        f.write('\n')
                        xywhn = list(xywhn)
                        xywhn.append('c1')
                        boxes_xywhn.append(xywhn)

                uber_augmenter_bb(im, boxes_xywhn, classes, self.yolo_patch_path, 'grid', 5, do_horizontal_flip=do_horizontal_flip, do_vertical_flip=do_vertical_flip,
                                  do_random_rotate90=do_random_rotate90, do_random_sized_crop=do_random_sized_crop, do_random_brightness_contrast=do_random_brightness_contrast,
                                  do_random_gamma=do_random_gamma, do_color_jitter=do_color_jitter)

        print()

    def set_pretrained_model(self, pretrained_model_path, model_type):
        if model_type == DLModel.STARDIST:
            self.models[DLModel.STARDIST] = StardistInstanceModel(self.patch_path, self.model_path, self.num_classes, pretrained_model_path)
        elif model_type == DLModel.YOLO_SAM:
            self.models[DLModel.YOLO_SAM] = YoloSAMModel(None, self.model_path, self.num_classes, pretrained_model_path)

    def get_model(self, network_type):
        if self.models[network_type] is None:
            if network_type == DLModel.UNET:
                self.models[network_type] = PytorchSemanticModel(self.patch_path, self.model_path, self.num_classes)
            elif network_type == DLModel.STARDIST:
                self.models[network_type] = StardistInstanceModel(self.patch_path, self.model_path, self.num_classes)
            elif network_type == DLModel.CELLPOSE:
                self.models[network_type] = CellPoseInstanceModel(self.patch_path, self.model_path, self.num_classes)
            elif network_type == DLModel.YOLO_SAM:
                self.models[network_type] = YoloSAMModel(self.patch_path, self.model_path, self.num_classes)
            elif network_type == DLModel.MOBILE_SAM2:
                self.models[network_type] = MobileSAMModel(self.patch_path, self.model_path)
        return self.models[network_type]

    def perform_training(self, network_type, num_epochs, update):

        if update is not None:
            update(f"Training {network_type} model...", 0)
        
        model = self.get_model(network_type) 
        model.train(num_epochs, update)

    def predict(self, n, network_type, update):
        
        image = self.image_list[n]
        
        if network_type == DLModel.YOLO_SAM or network_type == DLModel.MOBILE_SAM2:
            if update is not None:
                update(f"Apply {network_type} to image "+str(n)+"...")

            model = self.get_model(network_type)
            prediction, results = model.predict(image)
            boxes = self.xyxy_to_tltrbrbl(results, n)
            return prediction, boxes
        else:
            prediction = self.get_model(network_type).predict(image)
            return prediction
        
    def predict_all(self, network_type, update):
        predictions = []
        
        if network_type == DLModel.YOLO_SAM or network_type == DLModel.MOBILE_SAM2:
            predictions = []
            boxes = []
            for z in range(len(self.image_list)):
                if update is not None:
                    update(f"Apply {network_type} to image "+str(z)+"...")
                print('predicting image ', z)
                
                model = self.get_model(network_type)

                image = self.image_list[z]
                prediction, result = model.predict(image)
                
                predictions.append(prediction)

                boxes_ = self.xyxy_to_tltrbrbl(result, z)
                boxes = boxes + boxes_

            return predictions, boxes
        else:
            for z in range(len(self.image_list)):
                image = self.image_list[z]
                model = self.get_model(network_type)
                prediction = model.predict(image)
                predictions.append(prediction)
            return predictions
    
    def xyxy_to_tltrbrbl(self, boxes, n):
        boxes_ = []
        for box in boxes:
            tl = [n, box[1], box[0]]
            tr = [n, box[1], box[2]]
            br = [n, box[3], box[2]]
            bl = [n, box[3], box[0]]
            bbox = [tl, tr, br, bl]
            boxes_.append(bbox)
        return boxes_
      