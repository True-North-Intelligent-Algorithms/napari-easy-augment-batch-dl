from pathlib import Path
import os
from skimage.io import imread, imsave
import numpy as np
from tnia.deeplearning.dl_helper import generate_patch_names, generate_next_patch_name
import json
from tnia.deeplearning.dl_helper import make_label_directory
from tnia.deeplearning.augmentation import uber_augmenter
from tnia.deeplearning.dl_helper import quantile_normalization

class DeepLearningProject:
    def __init__(self, parent_path, num_classes):

        self.parent_path = Path(parent_path)
        self.image_folder = Path(parent_path)

        self.image_path = Path(parent_path)
        self.label_path = Path(parent_path / r'labels')
        self.napari_path = Path(parent_path / r'napari')

        self.patch_path= self.parent_path / 'patches'
        self.model_path = self.parent_path / 'models'

        if not os.path.exists(self.patch_path):
            os.mkdir(self.patch_path)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.num_classes = num_classes
        
        self.files = list(self.image_path.glob('*.jpg'))
        self.files = self.files+list(self.image_path.glob('*.tif'))
        
        self.image_label_paths, self.mask_label_paths = make_label_directory(1, self.num_classes, self.label_path)

        #self.image_files = self.load_image_files()
        if self.napari_path.exists():
            self.image_list = []
            for index in range(len(self.files)):

                im = imread(self.files[index])
                print(im.shape)
                self.image_list.append(im)

            self.images = np.load(self.napari_path / 'images.npy')

            self.label_list = []
            for c in range(num_classes):
                self.label_list.append(np.load(os.path.join(self.napari_path, 'labels_'+str(c)+'.npy')))

            self.boxes = np.load(self.napari_path / 'Label Box.npy')
            
          
        else:
            self.initialize_napari_project()

    def initialize_napari_project(self):
        print('Napari directory does not exist, initializing project')



        self.image_list = []
        self.label_list = []

        for c in range(self.num_classes):
            self.label_list.append([])

        for index in range(len(self.files)):

            im = imread(self.files[index])
            print(im.shape)
            self.image_list.append(im)

            for c in range(self.num_classes): 
                self.label_list[c].append(np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8))

        self.images = self.pad_to_largest(self.image_list) #np.array(self.image_list)
        #self.images = np.array(self.image_list)

        for c in range(self.num_classes):
            self.label_list[c] =  self.pad_to_largest(self.label_list[c])#np.array(self.label_list[c])
            #self.label_list[c] = np.array(self.label_list[c])

        self.boxes = None

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

    def save_project(self, boxes, layers=None):
        
        if not self.napari_path.exists():
            os.makedirs(self.napari_path)
        
        if layers is not None:
            for layer in layers:
                np.save(os.path.join(self.napari_path, layer.name+'.npy'), layer.data)

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

    def perform_augmentation(self, boxes, num_patches = 100, patch_size=256):

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
        
       