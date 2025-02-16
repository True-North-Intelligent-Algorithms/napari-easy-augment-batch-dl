# Dependencies

Napari-Easy-Augment-Batch-DL relies on one or more deep learning frameworks being installed in your environment.  On start-up it attempts to import the following deep learning toolkits.  At least one has to be successfully imported

**Cellpose:**  For instance segmentation  
**Stardist:**  For instance segmentation  
**Pytorch:**  For semantic segmentation  
**Mobile SAM:**  An approach that uses Yolo + SAM for instance segmentation. 

Other frameworks can be added via our plugin mechanism (Documentation to come)

Below are sets of suggested installation instructions for Linux, Mac M1, and Windows.  Not all dependencies are needed.  You need atleast one deep learning framework, but for example if you would like to work with Cellpose only you do not need Stardist or SAM. 

## Linux

```
    conda create -n I2K2024_dl python=3.10
    conda activate I2K2024_dl
    pip install numpy==1.26 # 
    pip install "napari[all]" # requires quotation marks to parse the square brackets on Linux, not sure if generalizes
    pip install albumentations
    pip install matplotlib
    pip install "tensorflow[and-cuda]" # as above, requires quotation marks
    pip install stardist 
    pip install gputools==0.2.15 # I think numpy v2.1.2 gets installed here for me
    pip install edt # pip throws: numba v0.60.0, tensorflow v2.18.0 require lower versions of numpy. Should still work but iffy.
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/segment-everything.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything.git
```

## Mac M1

```
    conda create -n I2K2024_dl python=3.10
    conda activate I2K2024_dl
    pip install numpy==1.26 # 
    pip install "napari[all]" # also requires quotes on Mac
    pip install "tensorflow[and-cuda]" # as above, requires quotation marks. Pip throws a bunch of conflicts, but whatever.
    pip install stardist
    pip install gputools==0.2.15 # pip dependency incompatibilities
    pip install edt 
    pip install torch torchvision torchaudio # remove the index flag url
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/segment-everything.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything.git
```

## Windows 

Note that to use GPU with Stardist in Windows we need two different environments because Windows needs an older version of tensorflow, which is not compatible wither newer version of pytorch based toolkits. 

### Windows stardist environment

```
    conda create -n I2k2024_stardist python=3.10
    conda activate I2k2024_stardist
    pip install numpy==1.26
    pip install "napari[all]"
    pip install albumentations
    pip install matplotlib
    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.1.0
    pip install "tensorflow<2.11"
    pip install stardist 
    pip install gputools==0.2.15 
    pip install edt
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
```

### Windows pytorch environment

```
    conda create -n I2k2024_pytorch python=3.10
    conda activate I2k2024_pytorch
    pip install numpy==1.26
    pip install "napari[all]"
    pip install albumentations
    pip install matplotlib
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
    pip install cellpose
    pip install git+https://github.com/True-North-Intelligent-Algorithms/segment-everything.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-easy-augment-batch-dl.git
    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git 
    pip install git+https://github.com/True-North-Intelligent-Algorithms/napari-segment-everything.git
```
