import numpy as np
from sympy import im
from napari_easy_augment_batch_dl.frameworks.base_framework import BaseFramework, LoadMode
from tifffile import imread
import json
from napari_easy_augment_batch_dl.frameworks.pytorch_semantic_dataset import PyTorchSemanticDataset
from torch.utils.data import DataLoader
from monai.networks.nets import BasicUNet
import torch
from tnia.deeplearning.dl_helper import quantile_normalization
from torchvision import transforms
from torchvision.transforms import v2
import os
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class PytorchSemanticFramework(BaseFramework):

    semantic_thresh: float = field(metadata={'type': 'float', 'harvest': True, 'advanced': False, 'training': False, 'min': -10.0, 'max': 10.0, 'default': 0.0, 'step': 0.1})
    num_classes: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': False, 'min': 0, 'max': 100, 'default': 2, 'step': 1})
    
    sparse: bool = field(metadata={'type': 'bool', 'harvest': True, 'advanced': False, 'training': True, 'default': True})
    num_epochs: int = field(metadata={'type': 'int', 'harvest': True, 'advanced': False, 'training': True, 'min': 0, 'max': 100000, 'default': 100, 'step': 1})
    model_name: str = field(metadata={'type': 'str', 'harvest': True, 'advanced': False, 'training': True, 'default': 'semantic', 'step': 1})
    
    def __init__(self, parent_path: str,  num_classes: int, start_model: str = None):
        super().__init__(parent_path, num_classes)
        
        self.model = None
        '''
        # get path from model_name
        if os.path.isdir(model_name):
            self.model_path = model_name
            self.model = None
            #model_name = os.path.join(model_path, 'model.pth')
        else:
            self.model_path = os.path.dirname(model_name)
            self.model = torch.load(model_name )
        '''

        self.model_name = self.generate_model_name(
            base_name="model"
        )
        
        #super().__init__(patch_path, self.model_path, num_classes)

        #self.threshold = 0.5
        self.semantic_thresh = 0.0
        self.descriptor = "Pytorch Semantic Model"
        self.load_mode = LoadMode.File
        self.sparse = True
        self.num_epochs = 100
        self.num_classes = 2 
        
        self.model_name = self.generate_model_name('semantic')

    def generate_model_name(self, base_name="model"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
        
    def create_callback(self, updater):
        self.updater = updater
        pass
    
    def train(self, updater=None):

        patch_path = Path(self.patch_path)
        
        if updater is None:
            updater = self.updater
        
        if updater is not None:
            updater('Training Pytorch Semantic model', 0)

        cuda_present = torch.cuda.is_available()
        ndevices = torch.cuda.device_count()
        use_cuda = cuda_present and ndevices > 0
        device = torch.device("cuda" if use_cuda else "cpu")  # "cuda:0" ... default device, "cuda:1" would be GPU index 1, "cuda:2" etc
        print("number of devices:", ndevices, "\tchosen device:", device, "\tuse_cuda=", use_cuda)

        with open(patch_path / 'info.json', 'r') as json_file:
            data = json.load(json_file)
            sub_sample = data.get('sub_sample',1)
            print('sub_sample',sub_sample)
            axes = data['axes']
            print('axes',axes)
            num_inputs = data['num_inputs']
            print('num_inputs',num_inputs)
            num_truths = data['num_truths']
            print('num_truths',num_truths)


        image_patch_path = patch_path / 'input0'
        label_patch_path = patch_path / 'ground truth0'

        from glob import glob

        tif_files = glob(str(image_patch_path / '*.tif'))

        first_im = imread(tif_files[0])
        target_shape=first_im.shape

        print('target_shape',target_shape)

        if axes == 'YX':
            num_in_channels=1
        else:
            num_in_channels=3

        assert patch_path.exists(), f"root directory with images and masks {patch_path} does not exist"

        X = sorted(patch_path.rglob('**/input0/*.tif'))

        Y = []
        for i in range(num_truths):
            Y.append(sorted(patch_path.rglob(f'**/ground truth{i}/*.tif')))

        train_data = PyTorchSemanticDataset(
            image_files=X,
            label_files_list=Y,
            target_shape=target_shape
        )

        # NOTE: the length of the dataset might not be the same as n_samples
        #       because files not having the target shape will be discarded
        print(len(train_data))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        # there is an inconstency in how different classes can be defined
        # 1. every class has it's own label image
        # 2. every class has a unique value in the label image
        # When I wrote a lot of this code I was thinking of the first case, but now see the second may be easier for the user
        # so number of output channels is the max of the truth image

        if self.model == None:
            self.model = BasicUNet(
                spatial_dims=2,
                in_channels=num_in_channels,
                out_channels=self.num_classes,
                #features=[16, 16, 32, 64, 128, 16],
                act="softmax",
                #norm="batch",
                dropout=0.25,
            )

        # Important: transfer the model to the chosen device
        self.model = self.model.to(device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.e-3)
        init_params = list(self.model.parameters())[0].clone().detach()

        log_interval = 20
        self.model.train(True)

        # BCEWithLogitsLoss combines sigmoid + BCELoss for better
        # numerical stability. It expects raw unnormalized scores as input which are shaped like 
        # B x C x W x D
        #loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=-1)

        for epoch in range(1, self.num_epochs + 1):
            for batch_idx, (X, y) in enumerate(train_loader):
                # the inputs and labels have to be on the same device as the model
                if self.sparse:
                    #y = y.astype(np.int64)
                    y = y-1

                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()

                prediction_logits = self.model(X)
                
                y = torch.squeeze(y,1)
                batch_loss = loss_function(prediction_logits, y)

                batch_loss.backward()

                optimizer.step()

                if batch_idx % log_interval == 0:
                    print(
                        "Train Epoch:",
                        epoch,
                        "Batch:",
                        batch_idx,
                        "Total samples processed:",
                        (batch_idx + 1) * train_loader.batch_size,
                        "Loss:",
                        batch_loss.item(),
                    )
                    if updater is not None:
                        progress = int(epoch/self.num_epochs*100)
                        updater(f"Epoch {epoch} Batch {batch_idx} Loss {batch_loss.item()}", progress)
        torch.save(self.model, Path(self.model_path) / self.model_name)

    def predict(self, image):
        device = torch.device("cuda")
        image_ = image.copy().astype(np.float32)

        axes = 'YCX'

        if axes == 'YXC':
            for i in range(1):
                image_[:,:,i] = quantile_normalization(
                    image_[:,:,i],
                    quantile_low=0.01,
                    quantile_high=0.998,
                    clip=True).astype(np.float32)
        else:
            image_ = quantile_normalization(
                image_,
                quantile_low=0.01,
                quantile_high=0.998,
                clip=True).astype(np.float32)

        tensor_transform = transforms.Compose([
            v2.ToTensor(),
        ])
        x = tensor_transform(image_)
        x = x.unsqueeze(0).to(device)
        #x = torch.from_numpy(testim_).to(device)

        print(x.shape)
        self.model.eval()
        
        #with torch.no_grad():
        #    y = self.model(x)

        # here we chunk the input into 4 parts to avoid running out of memory
        # on some systems this is needed on some it is not.  Sometimes we may
        # even need smaller (more) chunks... so this is a bit of a WIP.
        outputs = []
        for chunk in torch.chunk(x, chunks=4, dim=3):  # Divide input into smaller parts
            
            with torch.no_grad():
                outputs.append(self.model(chunk))
            del chunk
            torch.cuda.empty_cache()
        y = torch.cat(outputs, dim=3)

        prediction = y.cpu().detach()[0, 0].numpy()
        prediction = y.cpu().detach().numpy()
        prediction = np.squeeze(prediction)

        c1 = (prediction[0]> prediction[1]) & (prediction[0]> prediction[2])
        c2 = (prediction[1]> prediction[0]) & (prediction[1]> prediction[2])
        c3 = (prediction[2]> prediction[0]) & (prediction[2]> prediction[1])

        prediction = c1+2*c2+3*c3

        return prediction

    def load_model_from_disk(self, model_name):
        self.model = torch.load(model_name)
        base_name = os.path.basename(model_name)
        self.model_dictionary[base_name] = self.model

BaseFramework.register_framework('PytorchSemanticFramework', PytorchSemanticFramework)


