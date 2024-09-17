from pathlib import Path
import os
from skimage.io import imread, imsave
import numpy as np
from tnia.deeplearning.dl_helper import generate_patch_names, generate_next_patch_name
import json
from tnia.deeplearning.dl_helper import make_label_directory
from tnia.deeplearning.augmentation import uber_augmenter, uber_augmenter_bb
from tnia.deeplearning.dl_helper import quantile_normalization
from napari_easy_augment_batch_dl.bounding_box_util import is_bbox_within, tltrblbr_to_normalized_xywh, normalized_xywh_to_tltrblbr, x1y1x2y2_to_tltrblbr, yolotxt_to_naparibb
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
        self.patch_path= self.parent_path / 'patches'
        self.model_path = self.parent_path / 'models'

        self.annotation_path = self.parent_path / 'annotations'
        self.prediction_path = self.parent_path / 'predictions'

        # paths for yolo labels
        self.yolo_label_path = Path(parent_path / r'yolo_labels')
        self.yolo_patch_path = Path(parent_path / r'yolo_patches')
        self.yolo_image_label_paths = [os.path.join(self.yolo_label_path, 'images')]
        self.yolo_mask_label_paths = [os.path.join(self.yolo_label_path, 'labels')]

        self.yolo_predictions = Path(parent_path / r'yolo_predictions')

        if not os.path.exists(self.patch_path):
            os.mkdir(self.patch_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if not os.path.exists(self.yolo_label_path):
            os.mkdir(self.yolo_label_path)
        if not os.path.exists(self.yolo_patch_path):
            os.mkdir(self.yolo_patch_path)
        if not os.path.exists(self.annotation_path):
            os.mkdir(self.annotation_path)
        if not os.path.exists(self.prediction_path):
            os.mkdir(self.prediction_path)

        if not os.path.exists(self.yolo_image_label_paths[0]):
            os.mkdir(self.yolo_image_label_paths[0])
        if not os.path.exists(self.yolo_mask_label_paths[0]):
            os.mkdir(self.yolo_mask_label_paths[0])

        if not os.path.exists(self.yolo_predictions):
            os.mkdir(self.yolo_predictions)
                    
        self.files = list(self.image_path.glob('*.jpg'))
        self.files = self.files+list(self.image_path.glob('*.jpeg'))
        self.files = self.files+list(self.image_path.glob('*.tif'))
        self.files = self.files+list(self.image_path.glob('*.tiff'))
        self.files = self.files+list(self.image_path.glob('*.png'))
        
        self.image_label_paths, self.mask_label_paths = make_label_directory(1, self.num_classes, self.label_path)
    
        self.image_list = []
        for index in range(len(self.files)):
            im = imread(self.files[index])
            if len(im.shape) == 3:
                if im.shape[2] > 3:
                    im = im[:,:,:3]
            self.image_list.append(im)
        
        self.label_list = []
        self.prediction_list = []
        self.annotation_list = []        
        self.boxes = []
        self.object_boxes = None
        self.predicted_object_boxes = None
        self.features = None
        self.predicted_features = None
        
        if os.path.exists(json_name):

            # load the label boxes.  
            for c in range(self.num_classes):
                labels_temp = []
                predictions_temp = []
                annotations_temp = []
                
                label_names = list(Path(self.mask_label_paths[c]).glob('*.tif'))
                json_names = list(Path(self.image_label_paths[0]).glob('*.json'))
                #print('there are {} labels for class {}'.format(len(label_names), c))
                #print('there are {} json files for class {}'.format(len(json_names), c))

                n=0            
                for image_name, image in zip(self.files, self.image_list):
                    image_base_name = image_name.name.split('.')[0]
                    
                    # get all label names for this image
                    label_names_ = [x for x in label_names if image_base_name in x.name]

                    # this code is fragile, if an image name is part of a another image name it will break
                    # !  TODO:  REWORK
                    # get all json names for this image
                    json_names_ = [x for x in json_names if image_base_name in x.name]

                    # sort the label names and json names to make sure they correspond
                    label_names_ = sorted(label_names_)
                    json_names_ = sorted(json_names_)

                    label = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

                    #print('image base name is ', image_base_name)

                    for label_name_, json_name_ in zip(label_names_, json_names_):

                        #print('label name is ', label_name_)
                        #print('json name is ', json_name_)

                        with open(json_name_, 'r') as f:
                            json_ = json.load(f)
                            #print(json_)
                                                        
                            x1= json_['bbox'][0]
                            y1= json_['bbox'][1]
                            x2= json_['bbox'][2]
                            y2= json_['bbox'][3]

                            bbox = x1y1x2y2_to_tltrblbr(x1, y1, x2, y2, n)
                            self.boxes.append(bbox)

                            label_crop = imread(label_name_)
                            label[y1:y2, x1:x2] = label_crop
                            #rois.append([[x1, y1], [x2, y2]])

                    labels_temp.append(label)

                    annotation_name = self.get_annotation_name(n, c)
                    # check if annotation_name exists
                    if os.path.exists(annotation_name):
                        annotation = imread(annotation_name)
                    else:
                        annotation = label #np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

                    # check if prediction_name exists and if so load it
                    prediction_name = self.get_prediction_name(n, c)
                    if os.path.exists(prediction_name):
                        prediction = imread(prediction_name)
                    else:
                        prediction = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                        
                    predictions_temp.append(prediction)
                    annotations_temp.append(annotation)
                    n = n+1


                self.label_list.append(labels_temp)
                self.prediction_list.append(predictions_temp)
                self.annotation_list.append(annotations_temp)                        
            
            # now load the yolo labels (bounding boxes)

            self.yolo_image_label_paths = [os.path.join(self.yolo_label_path, 'images')]
            self.yolo_mask_label_paths = [os.path.join(self.yolo_label_path, 'labels')]
            
            self.object_boxes = []
            self.predicted_object_boxes = []
            
            self.features = pd.DataFrame(columns=['class'])            

            # loop loading the yolo bounding boxes.  For each image the yolo bounding boxes are stored in a text file 
            n=0
            for image_name, image in zip(self.files, self.image_list):
                
                image_base_name = image_name.name.split('.')[0]

                # get the name of the the yolo format bounding boxes text file
                yolo_txt_name = os.path.join(self.yolo_mask_label_paths[0], image_base_name+'.txt')

                # if the text file doesn't exist this image does not have bounding boxes
                if not os.path.exists(yolo_txt_name):
                    n = n+1
                    continue

                object_boxes, features = yolotxt_to_naparibb(yolo_txt_name, image.shape, n)
                self.object_boxes = self.object_boxes + object_boxes
                self.features = pd.concat([self.features, features], ignore_index=True)
                ''''                   
                # load yolo txt file
                with open(yolo_txt_name, 'r') as f:
                    lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        parts = line.split(' ')
                        
                        class_ = int(parts[0])
                        
                        # get the normalized x center, y center, width, height
                        xywhn = [float(x) for x in parts[1:5]]
                        xywhn = np.array(xywhn)

                        # convert to top left, top right, bottom left, bottom right pixel coordinates
                        xyxy = normalized_xywh_to_tltrblbr(xywhn[0], xywhn[1], xywhn[2], xywhn[3], image.shape[1], image.shape[0])
                        xyxy = [[n, xyxy[0][0], xyxy[0][1]], [n, xyxy[1][0], xyxy[1][1]], [n, xyxy[2][0], xyxy[2][1]], [n, xyxy[3][0], xyxy[3][1]]]
                        
                        # add to the bounding box list
                        self.object_boxes.append(np.array(xyxy))
                        
                        # add the class to a data frame
                        # TODO: this format is useful for napari, but it make make sense to refactor this to the 
                        # napari specific 'easy_augment_batch_dl' class
                        df_new = pd.DataFrame([{'class': class_}])
                        self.features = pd.concat([self.features,df_new], ignore_index=True) 
                '''
                n = n+1

            self.object_boxes = np.array(self.object_boxes)

        else:
            for c in range(self.num_classes):
                self.label_list.append([])
                self.prediction_list.append([])
                self.annotation_list.append([])
                
                for image in self.image_list:
                    self.label_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))
                    self.prediction_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))
                    self.annotation_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))

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

    def save_project(self, boxes, labels_from_napari):

        # save json file with num_classes
        json_name = os.path.join(self.parent_path, 'info.json')

        with open(json_name, 'w') as f:
            json_ = {}
            json_['num_classes'] = self.num_classes
            json.dump(json_, f)

        # delete old labels TODO: think this over could be dangerous if something goes wrong with resaving
        self.delete_all_files_in_directory(self.image_label_paths[0])
        for c in range(self.num_classes):
            self.delete_all_files_in_directory(self.mask_label_paths[c])

        # start a dataframe to store the bounding boxes
        df_bounding_boxes = pd.DataFrame(columns=['file_name', 'xstart', 'ystart', 'xend', 'yend'])            

        # loop through all label bounding box saving the image and label data for the bounding box        
        for box in boxes:
            z = int(box[0,0])

            name = self.files[z].name.split('.')[0]

            # get rid of first column (z axis)
            box = box[:,1:]

            ystart = int(np.min(box[:,0]))
            yend = int(np.max(box[:,0]))
            xstart = int(np.min(box[:,1]))
            xend = int(np.max(box[:,1]))

            # add this bounding box to the dataframe
            df_temp = pd.DataFrame([{'file_name': self.files[z].name, 'xstart': xstart, 'ystart': ystart, 'xend': xend, 'yend': yend}])
            df_bounding_boxes = pd.concat([df_bounding_boxes,df_temp], ignore_index=True) 

            #print('bounding box is',ystart, yend, xstart, xend)
            print('image file is ', self.files[z])

            if np.ndim(self.image_list[z]) == 3:
                im = self.image_list[z][ystart:yend, xstart:xend, :]
            else:
                im = self.image_list[z][ystart:yend, xstart:xend]

            labels=[]
            
            for c in range(self.num_classes):
                #labels.append(self.label_list[c][z][ystart:yend, xstart:xend])
                labels.append(labels_from_napari[c][z,ystart:yend, xstart:xend])
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

        # save annotations and predictions
        z = 0
        for image in self.image_list:
            for c in range(self.num_classes):
                height, width = image.shape[:2]
                annotation = labels_from_napari[c][z][:height, :width]
                annotation = annotation.astype(np.uint16)
                
                annotation_name = self.get_annotation_name(z, c)
                imsave(annotation_name, annotation)

                prediction = self.prediction_list[c][z][:height, :width]
                prediction = prediction.astype(np.uint16)
                prediction_name = self.get_prediction_name(z, c)
                imsave(prediction_name, prediction)

            z = z + 1

        # save bouning box info to a csv file
        df_bounding_boxes.to_csv(os.path.join(self.label_path, 'training_labels.csv'), index=False)

    def get_annotation_name(self, z, c):
        annotation_class_dir = self.annotation_path / f'class_{c}'
        if not annotation_class_dir.exists():
            annotation_class_dir.mkdir(parents=True)
        base_name = self.files[z].name.split('.')[0]
        return annotation_class_dir / (base_name+'.tif')

    def get_prediction_name(self, z, c):
        prediction_class_dir = self.prediction_path / f'class_{c}'
        if not prediction_class_dir.exists():
            prediction_class_dir.mkdir(parents=True)
        base_name = self.files[z].name.split('.')[0]
        return prediction_class_dir / (base_name+'.tif')
    
    def save_object_boxes(self, object_boxes, object_classes):
        for object_box in object_boxes:
            print('object box is ', object_box)    
        
        object_boxes = np.array(object_boxes)

        num_images = len(self.image_list)

        print()

        for n in range(num_images):
            
            index_objects = np.all(object_boxes[:,:,0]==n, axis=1)
            filtered_object_boxes = object_boxes[index_objects]
            filtered_object_classes = object_classes[index_objects]

            if len(filtered_object_boxes) == 0:
                continue
            
            print(self.files[n])
            print(filtered_object_boxes)

            im = self.image_list[n]
            image_height = im.shape[0]
            image_width = im.shape[1]                
            
            # create text file for image n
            base_name = self.files[n].name.split('.')[0]
            yolo_name = os.path.join(self.yolo_mask_label_paths[0],(base_name + '.txt'))
            boxes_xywhn = []
            with open(yolo_name, 'w') as f:
                for object, class_ in zip(filtered_object_boxes, filtered_object_classes):
                    xywhn = tltrblbr_to_normalized_xywh(object[:,1:], image_width, image_height)
                    f.write(str(class_)+' ')
                    xywhn_str = str(xywhn).replace('(','').replace(')','').replace(',','')
                    f.write(xywhn_str)
                    f.write('\n')
                    xywhn = list(xywhn)
                    xywhn.append('c1')
                    boxes_xywhn.append(xywhn)
    
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

            ## IMPORTANT:  we apply normalization just before generating the patches
            #  this is useful because the normalization will be applied using the range of the 
            #  label values.  The labels are usually larger than the patches so normalization range
            # is calculated using a larger region.  This is better than normalizing on the smaller patches
            # Just be careful to match the quantile range used when predicting apr. to the range used here  
            im = quantile_normalization(im, quantile_low=0.003).astype(np.float32)

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
                        xywhn = tltrblbr_to_normalized_xywh(object[:,1:], image_width, image_height)
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
        elif model_type == DLModel.CELLPOSE:
            self.models[DLModel.CELLPOSE] = CellPoseInstanceModel(self.patch_path, self.model_path, self.num_classes, pretrained_model_path)
        elif model_type == DLModel.UNET:
            self.models[DLModel.UNET] = PytorchSemanticModel(self.patch_path, pretrained_model_path, self.num_classes)
            
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

    def predict(self, n, network_type, update, imagesz=1024):
        
        image = self.image_list[n]
            
        if update is not None:
            update(f"Apply {network_type} to image "+str(n)+"...")
    
        if network_type == DLModel.YOLO_SAM or network_type == DLModel.MOBILE_SAM2:

            model = self.get_model(network_type)
            prediction, results = model.predict(image, imagesz)
            boxes = self.xyxy_to_tltrbrbl(results, n)

            self.prediction_list[0][n] = prediction
            
            return prediction, boxes
        else:
            model = self.get_model(network_type)
            prediction = model.predict(image)

            self.prediction_list[0][n] = prediction

            return prediction
        
    def predict_all(self, network_type, update):

        if network_type == DLModel.YOLO_SAM or network_type == DLModel.MOBILE_SAM2:
            self.object_boxes = []
            self.predicted_object_boxes = []

            for c in range(self.num_classes):
                temp = [] 
                for n in range(len(self.image_list)):
                    prediction, boxes_ = self.predict(n, network_type, update)
                    temp.append(prediction)
                    self.object_boxes = self.object_boxes + boxes_
                    self.predicted_object_boxes = self.predicted_object_boxes + boxes_
                    
                self.prediction_list.append(temp)

        else:
            for c in range(self.num_classes):
                temp=[]
                for n in range(len(self.image_list)):
                    prediction = self.predict(n, network_type, update)
                    temp.append(prediction)
                self.prediction_list.append(temp)

    def set_cellpose_params(self, diameter, cellpose_channels = [0,1], cellprob_threshold = 0.0, flow_threshold = 0.4):
        cellpose_model = self.get_model(DLModel.CELLPOSE)
        cellpose_model.diameter = diameter
        cellpose_model.cellpose_channels = cellpose_channels
        cellpose_model.cellprob_threshold = cellprob_threshold
        cellpose_model.flow_threshold = flow_threshold    

    def set_stardist_params(self, prob_thresh, nms_thresh, scale):
        stardist_model = self.get_model(DLModel.STARDIST)
        stardist_model.prob_thresh = prob_thresh
        stardist_model.nms_thresh = nms_thresh
        stardist_model.scale = scale

    def set_pytorch_semantic_params(self, threshold):
        pytorch_model = self.get_model(DLModel.UNET)
        pytorch_model.threshold = threshold
    
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
      