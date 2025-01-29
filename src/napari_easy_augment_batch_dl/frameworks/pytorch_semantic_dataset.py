from tifffile import imread
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from tqdm import tqdm

class PyTorchSemanticDataset():

    def __init__(self, image_files, label_files_list, target_shape=(256, 256, 3)):
            """
            Parameters
            ----------
            image_files: list of pathlib.Path objects pointing to the *.tif images
            label_files_list: list of lists of pathlib.Path objects pointing to the *.tif segmentation masks
                        there are mulitple lists of label files each potentially representing one class
            target_shape: tuple of length 2 specifying the sample resolutions of files that
                        will be kept. All other files will NOT be used.
            """
            assert len(image_files) == len(label_files_list[0])
            assert all(x.name==y.name for x,y in zip(image_files, label_files_list[0]))

            self.images = []
            self.labels = []

            tensor_transform = transforms.Compose([
                v2.ToTensor(),
            ])

            # use tqdm to have eye pleasing error bars
            for idx in tqdm(range(len(image_files))):
                # we use the same data reading approach as in the previous notebook
                image = imread(image_files[idx])

                labels = []               
                for label_files in label_files_list:
                    label = imread(label_files[idx])
                    labels.append(label)
                
                # add channel dim if not present
                if len(image.shape) == 2:
                    image = np.expand_dims(image, axis=0)
                elif len(image.shape) == 3:
                    image = np.transpose(image, axes=(-1, *range(image.ndim - 1)))

                label = np.expand_dims(labels[0], axis=0)
                
                self.images.append(image)
                self.labels.append(label)

            # data is not a tensor yet but the Dataloader will handle that
            self.images = np.stack(self.images)
            self.labels = np.stack(self.labels).astype(np.int64)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)