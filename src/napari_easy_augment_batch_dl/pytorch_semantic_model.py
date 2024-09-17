import numpy as np
from napari_easy_augment_batch_dl.base_model import BaseModel
from tifffile import imread
import json
from napari_easy_augment_batch_dl.pytorch_semantic_dataset import PyTorchSemanticDataset
from torch.utils.data import DataLoader
from monai.networks.nets import BasicUNet
import torch
from tnia.deeplearning.dl_helper import quantile_normalization
from torchvision import transforms
from torchvision.transforms import v2
import os
from datetime import datetime

class PytorchSemanticModel(BaseModel):
    
    def __init__(self, patch_path, model_name, num_classes):
        # get path from model_name
        if os.path.isdir(model_name):
            self.model_path = model_name
            self.model = None
            #model_name = os.path.join(model_path, 'model.pth')
        else:
            self.model_path = os.path.dirname(model_name)
            self.model = torch.load(model_name )

        self.model_name = self.generate_model_name(
            base_name="model"
        )
        
        super().__init__(patch_path, self.model_path, num_classes)

        self.threshold = 0.5

    def generate_model_name(self, base_name="model"):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{base_name}_{current_time}.pth"
        return model_name
        
    def create_callback(self, updater):
        self.updater = updater
        pass
    
    def train(self, num_epochs, updater=None):
        
        if updater is None:
            updater = self.updater
        
        if updater is not None:
            updater('Training Pytorch Semantic model', 0)

        cuda_present = torch.cuda.is_available()
        ndevices = torch.cuda.device_count()
        use_cuda = cuda_present and ndevices > 0
        device = torch.device("cuda" if use_cuda else "cpu")  # "cuda:0" ... default device, "cuda:1" would be GPU index 1, "cuda:2" etc
        print("number of devices:", ndevices, "\tchosen device:", device, "\tuse_cuda=", use_cuda)

        with open(self.patch_path / 'info.json', 'r') as json_file:
            data = json.load(json_file)
            sub_sample = data.get('sub_sample',1)
            print('sub_sample',sub_sample)
            axes = data['axes']
            print('axes',axes)
            num_inputs = data['num_inputs']
            print('num_inputs',num_inputs)
            num_truths = data['num_truths']
            print('num_truths',num_truths)


        image_patch_path = self.patch_path / 'input0'
        label_patch_path = self.patch_path / 'ground truth0'

        from glob import glob

        tif_files = glob(str(image_patch_path / '*.tif'))

        first_im = imread(tif_files[0])
        target_shape=first_im.shape

        print('target_shape',target_shape)

        if axes == 'YX':
            num_in_channels=1
        else:
            num_in_channels=3

        assert self.patch_path.exists(), f"root directory with images and masks {self.patch_path} does not exist"

        X = sorted(self.patch_path.rglob('**/input0/*.tif'))

        Y = []
        for i in range(num_truths):
            Y.append(sorted(self.patch_path.rglob(f'**/ground truth{i}/*.tif')))

        train_data = PyTorchSemanticDataset(
            image_files=X,
            label_files_list=Y,
            target_shape=target_shape
        )

        # NOTE: the length of the dataset might not be the same as n_samples
        #       because files not having the target shape will be discarded
        print(len(train_data))

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

        if self.model == None:
            self.model = BasicUNet(
                spatial_dims=2,
                in_channels=num_in_channels,
                out_channels=num_truths,
                #features=[16, 16, 32, 64, 128, 16],
                act="relu",
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
        loss_function = torch.nn.BCEWithLogitsLoss(reduction="mean")

        for epoch in range(1, num_epochs + 1):
            for batch_idx, (X, y) in enumerate(train_loader):
                # the inputs and labels have to be on the same device as the model
                X, y = X.to(device), y.to(device)
                
                optimizer.zero_grad()

                prediction_logits = self.model(X)
                
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
                        progress = int(epoch/num_epochs*100)
                        updater(f"Epoch {epoch} Batch {batch_idx} Loss {batch_loss.item()}", progress)
        torch.save(self.model, self.model_path / self.model_name)

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
        y = self.model(x)

        prediction = y.cpu().detach()[0, 0].numpy()
        binary = prediction > self.threshold 

        return binary

    


