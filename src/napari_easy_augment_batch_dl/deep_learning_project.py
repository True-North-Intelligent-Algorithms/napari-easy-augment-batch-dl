from pathlib import Path
import os
from skimage.io import imread, imsave
import numpy as np
from tnia.deeplearning.dl_helper import (
    generate_patch_names, 
    generate_next_patch_name, 
    generate_next_name,
    zero_pad_index,
    make_label_directory, 
    quantile_normalization, 
    explicit_normalization, 
    compute_quantiles, 
    check_training_data
)
import json
from tnia.deeplearning.augmentation import uber_augmenter, uber_augmenter_bb

from napari_easy_augment_batch_dl.bounding_box_util import (
    tltrblbr_to_normalized_xywh,
    x1y1x2y2_to_tltrblbr,
    yolotxt_to_naparibb,
    tltrblbr_to_normalized_xywh,
    xyxy_to_tltrbrbl
)

import pandas as pd
import yaml
import glob 
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, TrainMode
from napari_easy_augment_batch_dl.zarr_helper import manage_zarr_store
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class ImageType(Enum):
    GRAY_SCALE_2D = "gray_scale_2d"
    GRAY_SCALE_3D = "gray_scale_3d"
    RGB_2D = "rgb_2d"
    RGB_3D = "rgb_3d"
    MULTI_CHANNEL_2D = "multi_channel_2d"
    MULTI_CHANNEL_3D = "multi_channel_3d"

class DeepLearningProject:
    def __init__(self, parent_path, num_classes=1):
        
        self.parent_path = Path(parent_path)

        # check if json.info exists
        json_name = self.parent_path / 'info.json'
        if os.path.exists(json_name):
            with open(json_name, 'r') as f:
                json_ = json.load(f)
                self.num_classes = json_['num_classes']
        else:    
            self.num_classes = num_classes

        self.image_path = self.parent_path
        self.label_path = self.parent_path / 'labels'
        self.patch_path= self.parent_path / 'patches'
        self.model_path = self.parent_path / 'models'

        self.annotation_path = self.parent_path / 'annotations'
        self.prediction_path = self.parent_path / 'predictions'

        # paths for yolo labels
        self.yolo_label_path = self.parent_path / 'yolo_labels'
        self.yolo_patch_path = self.parent_path / 'yolo_patches'
        self.yolo_image_label_paths = [self.yolo_label_path / 'images']
        self.yolo_mask_label_paths = [self.yolo_label_path / 'labels']
        self.yolo_predictions = self.parent_path / 'yolo_predictions'

        # path for machine learning
        self.ml_path = self.parent_path / 'ml'

        # make paths if they don't exist
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

        # collect all images in the image path
        # Note: this code loads all images in the directory.  This could be a problem if there are a large number of images                    
        self.image_file_list = list(self.image_path.glob('*.jpg'))
        self.image_file_list = self.image_file_list+list(self.image_path.glob('*.jpeg'))
        self.image_file_list = self.image_file_list+list(self.image_path.glob('*.tif'))
        self.image_file_list = self.image_file_list+list(self.image_path.glob('*.tiff'))
        self.image_file_list = self.image_file_list+list(self.image_path.glob('*.png'))

        self.image_file_list = sorted(self.image_file_list)
        
        self.image_label_paths, self.mask_label_paths = make_label_directory(1, self.num_classes, self.label_path)

        self.image_list = []
        for index in range(len(self.image_file_list)):
            im = imread(self.image_file_list[index])

            # if the image is 3D and the first dimension is less than 7 assume the first dimension is the channel
            # not really robust because you could have images with 6 slices... TODO:  REWORK
            if len(im.shape) == 3:
                # change channel to be last dimension       
                if im.shape[0]<=7:
                    im = np.transpose(im, (1,2,0))
            
                if im.shape[-1] > 3:
                    im = im[:,:,:3]
            
            self.image_list.append(im)
        
        self.prediction_list = []
        self.annotation_list = []        
        self.boxes = []
        self.object_boxes = None
        self.predicted_object_boxes = None
        self.classes = None
        self.predicted_classes = None

        self.frameworks = {} 

        # models derived from BaseFramework should be registered when imported
        # Here for each framework that has been registered we retrieve the class, instantiate it
        # and add it to the frameworks dictionary
        for framework_name in BaseFramework.registry:
            print('Found framework', framework_name)
            try:
                instance = BaseFramework.registry[framework_name](self.parent_path, self.num_classes)
                self.frameworks[instance.descriptor] = instance
            except Exception as e:
                print(f"Error instantiating class {framework_name}: {e}")

        # if there is already a json file this is a pre-existing project                
        if os.path.exists(json_name):

            # load the label boxes.  
            for c in range(self.num_classes):
                predictions_temp = []
                annotations_temp = []
                
                json_names = list(Path(self.image_label_paths[0]).glob('*.json'))

                n=0            
                for image_name, image in zip(self.image_file_list, self.image_list):
                    
                    name_without_extension = image_name.stem

                    # the goal now is to get all the labels and json files describing the bounding box of the label
                    # for this image.   

                    # use this array to collect all the json files that are associated with this image
                    json_names_ = []
                    
                    i = 0

                    # loop forming the json name, check if it exists, and if so add it to the list of json names  
                    # first loop uses zero padded index (this is the current way we name the labels so we have nice sorting)
                    while True:
                        next_name = name_without_extension + '_' + zero_pad_index(i)
                        json_name = os.path.join(self.image_label_paths[0], next_name + ".json")
                        
                        if not os.path.exists(json_name):
                            break
                        
                        json_names_.append(json_name)
                        i += 1 
                    
                    i = 0

                    # second loop uses non-zero padded index (this is the way we used to name the labels so we have to check for this too)
                    while True:
                        next_name = name_without_extension + '_' + str(i)
                        json_name = os.path.join(self.image_label_paths[0], next_name + ".json")
                        
                        if not os.path.exists(json_name):
                            break
                        
                        json_names_.append(json_name)
                        i += 1 
                    
                    # loop through all the jsons describing the bounding boxes and add them to the boxes list 
                    for json_name_ in json_names_:

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

                    # next add the annotations
                    # NOTE: there is a subtle difference between an annotation and label
                    # annotation: the image with object pixels labeled with an instance index
                    # label: the regions of the annotation that are marked with a label bounding box
                    # the idea is that we can use bounding boxes to quickly change which regions of the annotation are used for training
                    # end NOTE
                    annotation_name = self.get_annotation_name(n, c)

                    # check if annotation_name exists
                    if os.path.exists(annotation_name):
                        annotation = imread(annotation_name)
                    else:
                        annotation = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)

                    # check if prediction_name exists and if so load it
                    prediction_name = self.get_prediction_name(n, c)
                    if os.path.exists(prediction_name):
                        prediction = imread(prediction_name)
                    else:
                        prediction = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16)
                        
                    predictions_temp.append(prediction)
                    annotations_temp.append(annotation)
                    n = n+1

                self.prediction_list.append(predictions_temp)
                self.annotation_list.append(annotations_temp)                        
            
            # now load the yolo labels (bounding boxes)

            self.yolo_image_label_paths = [os.path.join(self.yolo_label_path, 'images')]
            self.yolo_mask_label_paths = [os.path.join(self.yolo_label_path, 'labels')]
            
            self.object_boxes = []
            self.predicted_object_boxes = []
            
            self.classes = pd.DataFrame(columns=['class'])            

            # loop loading the yolo bounding boxes.  For each image the yolo bounding boxes are stored in a text file 
            n=0
            for image_name, image in zip(self.image_file_list, self.image_list):
                
                name_without_extension = image_name.name.split('.')[0]

                # get the name of the the yolo format bounding boxes text file
                yolo_txt_name = os.path.join(self.yolo_mask_label_paths[0], name_without_extension+'.txt')

                # if the text file doesn't exist this image does not have bounding boxes
                if not os.path.exists(yolo_txt_name):
                    n = n+1
                    continue

                object_boxes, classes = yolotxt_to_naparibb(yolo_txt_name, image.shape, n)
                self.object_boxes = self.object_boxes + object_boxes
                self.classes = pd.concat([self.classes, classes], ignore_index=True)
                n = n+1

            self.object_boxes = np.array(self.object_boxes)
        # else this is a new project so just create a set of empty annotations and predictions
        else:
            for c in range(self.num_classes):
                self.prediction_list.append([])
                self.annotation_list.append([])
                
                for image in self.image_list:
                    self.prediction_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))
                    self.annotation_list[c].append(np.zeros((image.shape[0], image.shape[1]), dtype=np.uint16))

            for image_name in self.image_file_list:
                print(image_name)
            
        max_y = max(image.shape[0] for image in self.image_list)
        max_x = max(image.shape[1] for image in self.image_list)
        
        if len(self.image_list[0].shape) == 3:
            max_channels = max(image.shape[2] for image in self.image_list)
        else:
            max_channels = 1

        try:
            ml_labels_store = manage_zarr_store(os.path.join(self.ml_path,'ml_labels'), self.image_file_list, (max_y, max_x))
            ml_features_store = manage_zarr_store(os.path.join(self.ml_path,'ml_features'), self.image_file_list, (max_y, max_x, max_channels*12), dtype='f4')
            
            self.ml_labels = ml_labels_store['images']
            self.ml_features = ml_features_store['images']
        except Exception as e:
            print(f"Error creating ml labels and features: {e}")
        
        self.augmentation_parameters = {
            'size_factor': 1.25,
            'hue': 0.5,
            'brightness': 0.5,
            'saturation': 0.5,
            'alpha': 0.1,
            'sigma': 5,
            'alpha_affine': 5,
            'normalization_type': 'label',
        }

    # TODO: move to a utility class 
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

    def save_project(self, boxes, save_histograms=False):

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

        if save_histograms:
            hist_path = os.path.join(self.parent_path, 'histograms')

            if not os.path.exists(hist_path):
                os.mkdir(hist_path)
            
            # if saving the histograms loop through all images, save a copy, generate the histogram and save it
            # we save a copy of the image in the histogram directory to make it to inspect both the image and the histogram
            for image, image_name in zip(self.image_list, self.image_file_list):

                imsave(os.path.join(hist_path, image_name.name), image)

                # save the histogram of the image
                fig = Figure(figsize=(8, 6))
                ax = fig.add_subplot(111)

                ax.hist(image.flatten(), bins=100)
                ax.set_title('Histogram of image')
                ax.set_xlabel('Pixel value')
                ax.set_ylabel('Frequency')
                fig.savefig(os.path.join(hist_path, image_name.name.split('.')[0]+'_hist.png'))

        # loop through all label bounding box saving the image and label data for the bounding box        
        for box in boxes:

            # put a try around this because sometimes invalid boxes are generated
            try:
                z = int(box[0,0])

                name_without_extension = self.image_file_list[z].stem

                # get rid of first column (z axis)
                # TODO: refactor this because the z axis is only for Napari, getting rid of it should be done in the napari widget code
                box = box[:,1:]

                ystart = int(np.min(box[:,0]))
                yend = int(np.max(box[:,0]))
                xstart = int(np.min(box[:,1]))
                xend = int(np.max(box[:,1]))

                # add this bounding box to the dataframe
                df_temp = pd.DataFrame([{'file_name': self.image_file_list[z].name, 'xstart': xstart, 'ystart': ystart, 'xend': xend, 'yend': yend}])
                df_bounding_boxes = pd.concat([df_bounding_boxes,df_temp], ignore_index=True) 

                #print('bounding box is',ystart, yend, xstart, xend)
                print('image file is ', self.image_file_list[z])

                if np.ndim(self.image_list[z]) == 3:
                    im = self.image_list[z][ystart:yend, xstart:xend, :]
                else:
                    im = self.image_list[z][ystart:yend, xstart:xend]

                labels=[]
                
                for c in range(self.num_classes):
                    labels.append(self.annotation_list[c][z][ystart:yend, xstart:xend])
                    print('labelsum is ', labels[c].sum())

                print(im.shape, labels[0].shape)

                image_name, mask_name = generate_patch_names(str(self.image_label_paths[0]), str(self.mask_label_paths[0]), name_without_extension)
                base_name = generate_next_patch_name(str(self.image_label_paths[0]), name_without_extension)

                print(base_name)
                print(image_name)
                print(mask_name)

                # save the image corresponding to the label bounding box
                imsave(image_name, im)

                if save_histograms:
                    # also generate and save the histogram of the image 
                    hist_name = os.path.join(self.image_label_paths[0], base_name+"_hist.png")
                    
                    fig = Figure(figsize=(8, 6))
                    ax = fig.add_subplot(111)
                    ax.hist(im.flatten(), bins=100)
                    ax.set_title('Histogram of image')
                    ax.set_xlabel('Pixel value')
                    ax.set_ylabel('Frequency')
                    fig.savefig(hist_name)
                    plt.close(fig)  # Prevent notebooks from showing the figure

                print(image_name)

                # save the labels for each class
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
            except:
                print('error saving bounding box')

        # save annotations and predictions
        z = 0
        for image in self.image_list:
            for c in range(self.num_classes):
                height, width = image.shape[:2]
                annotation = self.annotation_list[c][z][:height, :width]
                annotation = annotation.astype(np.uint16)
                
                annotation_name = self.get_annotation_name(z, c)
                imsave(annotation_name, annotation)

                prediction = self.prediction_list[c][z][:height, :width]
                prediction = prediction.astype(np.uint16)
                prediction_name = self.get_prediction_name(z, c)
                imsave(prediction_name, prediction)

            z = z + 1

        # save bounding box info to a csv file
        df_bounding_boxes.to_csv(os.path.join(self.label_path, 'training_labels.csv'), index=False)
        
    def get_file_name(self, path, z, c):
        class_dir = path / f'class_{c}'
        if not class_dir.exists():
            class_dir.mkdir(parents=True)

        base_name = self.image_file_list[z].name.split('.')[0]
        full_base_name = class_dir / (base_name + '.tif')

        if full_base_name.exists():
            return full_base_name
        else:
            base_name = self.image_file_list[z].stem
            full_base_name = class_dir / (base_name + '.tif')
            return full_base_name 
        
    def get_annotation_name(self, z, c):
        return self.get_file_name(self.annotation_path, z, c)

    def get_prediction_name(self, z, c):
        return self.get_file_name(self.prediction_path, z, c)
    
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
            
            print(self.image_file_list[n])
            print(filtered_object_boxes)

            im = self.image_list[n]
            image_height = im.shape[0]
            image_width = im.shape[1]                
            
            # create text file for image n
            base_name = self.image_file_list[n].name.split('.')[0]
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

            # get rid of first column (z axis)
            box = box[:,1:]

            ystart = int(np.min(box[:,0]))
            yend = int(np.max(box[:,0]))
            xstart = int(np.min(box[:,1]))
            xend = int(np.max(box[:,1]))

            if ystart < 0:
                ystart = 0
            if yend > self.image_list[z].shape[0]:
                yend = self.image_list[z].shape[0]
            if xstart < 0:
                xstart = 0
            if xend > self.image_list[z].shape[1]:
                xend = self.image_list[z].shape[1]

            if np.ndim(self.image_list[z]) == 3:
                im = self.image_list[z][ystart:yend, xstart:xend, :]
            else:
                im = self.image_list[z][ystart:yend, xstart:xend]

            labels=[]

            
            for c in range(self.num_classes):
                labels.append(self.annotation_list[c][z][ystart:yend, xstart:xend])

            ## IMPORTANT:  we apply normalization just before generating the patches
            #  this is useful because the normalization will be applied using the range of the 
            #  label values.  The labels are usually larger than the patches so normalization range
            # is calculated using a larger region.  This is better than normalizing on the smaller patches
            # Just be careful to match the quantile range used when predicting apr. to the range used here
            # TODO: need to make this a parameter  
            im = quantile_normalization(im).astype(np.float32)

            #do_random_sized_crop=True, do_random_brightness_contrast=True, 
            #do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False)
            
            #  Here we get rid of the alpha channel if it exists... but logic won't work for 3D images.
            #  TODO: Revisit as part of the 3D image work
            if len(im.shape) == 3:
                if im.shape[-1]>3:
                    im = im[:,:,:3]
                
            name = self.image_file_list[z].name.split('.')[0]
            
            uber_augmenter(im, labels, patch_path, name, patch_size, num_patches, do_vertical_flip, do_horizontal_flip, do_random_rotate90, do_random_sized_crop, do_random_brightness_contrast, 
                                  do_random_gamma, do_color_jitter, do_elastic_transform, 1, **self.augmentation_parameters)
       
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
                
                imsave(os.path.join(self.yolo_image_label_paths[0], self.image_file_list[n].name), im)

                # create text file for image n
                base_name = self.image_file_list[n].name.split('.')[0]
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

    def get_model(self, network_type):
        return self.frameworks[network_type]
    
    def check_patches(self):
        return check_training_data(self.patch_path)

    def check_label(self, n, index):
        
        im = self.image_list[n][index]

        label_good = False
            
        for c in range(self.num_classes):
            label = self.annotation_list[c][n][index]
            sum = label.sum()
            if sum>0:
                label_good = True
            print('labelsum is ', sum)
        
        return label_good

    def check_labels(self, boxes):
        # loop through all boxes and check if we have pixel data drawn...
        # (a bit of a question is what we do here... sometimes empty labels may be valid when training a model to 'not learn')
        # (that being said at least some of the boxes should have labels)
        for box in boxes:  
            
            # sequence number of box
            n = int(box[0,0])
                
            # TODO: refactor this because the z axis is only for Napari, getting rid of it should be done in the napari widget code
            box = box[:,1:]
            print(box)

            ystart = int(np.min(box[:,0]))
            yend = int(np.max(box[:,0]))
            xstart = int(np.min(box[:,1]))
            xend = int(np.max(box[:,1]))

            index = np.s_[ystart:yend, xstart:xend]

            label_good = self.check_label(n, index) 
            print(label_good) 
            print() 

        return True



    def perform_training(self, network_type, update):

        if update is not None:
            update(f"Training {network_type} model...", 0)
        
        model = self.get_model(network_type)
        model.train(update)

    def predict_roi(self, n, network_type, update, roi):
        image = self.image_list[n][roi]
        model = self.frameworks[network_type]

        if update is not None:
            update(f"Apply {network_type} to image "+str(n)+"...")

        prediction = model.predict(image)

        return prediction
         
    def predict(self, n, network_type, update):
        
        image = self.image_list[n]
            
        if update is not None:
            update(f"Apply {network_type} to image "+str(n)+"...")
        
        model = self.frameworks[network_type]

        if model.boxes == True:

            model = self.get_model(network_type)
            prediction, results = model.predict(image)
            boxes = xyxy_to_tltrbrbl(results, n)

            self.prediction_list[0][n] = prediction
            
            return prediction, boxes
        else:
            prediction = model.predict(image)

            self.prediction_list[0][n] = prediction

            return prediction
            
    def predict_all(self, network_type, update):
        model = self.frameworks[network_type]
        
        if model.boxes == True:
            self.object_boxes = []
            self.predicted_object_boxes = []

            for c in range(self.num_classes):
                temp = [] 
                for n in range(len(self.image_list)):
                    prediction, boxes_ = self.predict(n, network_type, None)
                    progress = int((n/len(self.image_list))*100)
                    update(f"Apply {network_type} to image "+str(n), progress)
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
    
    def set_augmentation_parameter(self, param_name, value):
        """ Set the augmentation parameter for the framework """
        self.augmentation_parameters[param_name] = value
   
     