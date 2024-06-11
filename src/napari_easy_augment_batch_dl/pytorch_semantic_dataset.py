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

                if image.shape != target_shape:
                    continue
                
                # NOTE: we convert the label to dtype float32 and not uint8 because
                # the tensor transformation does a normalization if the input is of
                # dtype uint8, destroying the 0/1 labelling which we want to avoid.
                # label = fill_label_holes(label)
                
                labels_binary = []

                for label in labels:
                    label_binary = np.zeros_like(label).astype(np.float32)
                    label_binary[label != 0] = 1.
                    labels_binary.append(label_binary)

                # convert to torch tensor: adds an artificial color channel in the front
                # and scales inputs to have same size as samples tend to differ in image
                # resolutions
                image = tensor_transform(image)
                labels_binary = np.stack(labels_binary, axis=2)
                label = tensor_transform(labels_binary)

                self.images.append(image)
                self.labels.append(label)

            self.images = torch.stack(self.images)
            self.labels = torch.stack(self.labels)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return len(self.images)