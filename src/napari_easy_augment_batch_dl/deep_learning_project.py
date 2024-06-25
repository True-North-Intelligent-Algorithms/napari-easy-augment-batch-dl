from pathlib import Path
import os
from skimage.io import imread, imsave
import numpy as np
from tnia.deeplearning.dl_helper import generate_patch_names, generate_next_patch_name
import json
from tnia.deeplearning.dl_helper import make_label_directory
from tnia.deeplearning.augmentation import uber_augmenter
from tnia.deeplearning.dl_helper import quantile_normalization
from napari_easy_augment_batch_dl.bounding_box_util import is_bbox_within, xyxy_to_normalized_xywh

class DeepLearningProject:
    def __init__(self, parent_path, num_classes=1):

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

        #self.image_files = self.load_image_files()
        if self.napari_path.exists():
            self.images = np.load(self.napari_path / 'images.npy')

            self.label_list = []
            for c in range(self.num_classes):
                self.label_list.append(np.load(os.path.join(self.napari_path, 'labels_'+str(c)+'.npy')))

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

    def initialize_napari_project(self):
        print('Napari directory does not exist, initializing project')

        self.label_list = []

        for c in range(self.num_classes):
            self.label_list.append([])

        for im in self.image_list:

            for c in range(self.num_classes): 
                self.label_list[c].append(np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8))

        self.images = self.pad_to_largest(self.image_list) #np.array(self.image_list)
        #self.images = np.array(self.image_list)

        for c in range(self.num_classes):
            self.label_list[c] =  self.pad_to_largest(self.label_list[c])#np.array(self.label_list[c])
            #self.label_list[c] = np.array(self.label_list[c])

        self.boxes = None
        self.object_boxes = None
        self.features = None
        
    def pad_to_largest(self, images):
        # Find the maximum dimensions
        max_rows = max(image.shape[0] for image in images)
        max_cols = max(image.shape[1] for image in images)
        
        # Create a list to hold the padded images
        padded_images = []
        
        for image in images:
            # Calculate the padding for each dimension
            pad_rows = max_rows - image.shape[0]
            pad_cols = max_cols - image.shape[1]
            
            if len(image.shape) == 3:
                # we occasionally hit rgba images, just use the first 3 channels
                image = image[:,:,:3]
                # Pad the array
                padded_image = np.pad(image, 
                                    ((0, pad_rows), (0, pad_cols), (0,0)), 
                                    mode='constant', 
                                    constant_values=0)
            else:
                padded_image = np.pad(image, 
                                    ((0, pad_rows), (0, pad_cols)), 
                                    mode='constant', 
                                    constant_values=0)
            
            padded_images.append(padded_image)
        
        # Stack the padded images along a new third dimension
        result = np.array(padded_images)
        
        return result

    def save_project(self, boxes):

        # save json file with num_classes
        json_name = os.path.join(self.parent_path, 'info.json')

        with open(json_name, 'w') as f:
            json_ = {}
            json_['num_classes'] = self.num_classes
            json.dump(json_, f)
        
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

            if np.ndim(self.images[z]) == 3:
                im = self.images[z,ystart:yend, xstart:xend, :]
            else:
                im = self.images[z,ystart:yend, xstart:xend]

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

    def perform_augmentation(self, boxes, num_patches = 100, patch_size=256, updater = None):

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

            if np.ndim(self.images[z]) == 3:
                im = self.images[z,ystart:yend, xstart:xend, :]
            else:
                im = self.images[z,ystart:yend, xstart:xend]

            labels=[]
            
            for c in range(self.num_classes):
                labels.append(self.label_list[c][z][ystart:yend, xstart:xend])

            im = quantile_normalization(im).astype(np.float32)
            uber_augmenter(im, labels, patch_path, 'grid', patch_size, num_patches, do_random_gamma=True, do_color_jitter = True)
       
    def perform_yolo_augmentation(self, boxes, objects, classes, num_patches_per_image, patch_size, updater=None):
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
            with open(yolo_name, 'w') as f:
                for object in filtered_objects:
                    xywhn = xyxy_to_normalized_xywh(object[:,1:], image_width, image_height)
                    f.write('0 ')
                    xywhn = str(xywhn).replace('(','').replace(')','').replace(',','')
                    f.write(xywhn)
                    f.write('\n')
        print()





       