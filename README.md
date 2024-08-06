# napari-easy-augment-batch-dl

[![License BSD-3](https://img.shields.io/pypi/l/napari-easy-augment-batch-dl.svg?color=green)](https://github.com/bnorthan/napari-easy-augment-batch-dl/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-easy-augment-batch-dl.svg?color=green)](https://pypi.org/project/napari-easy-augment-batch-dl)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-easy-augment-batch-dl.svg?color=green)](https://python.org)
[![tests](https://github.com/bnorthan/napari-easy-augment-batch-dl/workflows/tests/badge.svg)](https://github.com/bnorthan/napari-easy-augment-batch-dl/actions)
[![codecov](https://codecov.io/gh/bnorthan/napari-easy-augment-batch-dl/branch/main/graph/badge.svg)](https://codecov.io/gh/bnorthan/napari-easy-augment-batch-dl)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-easy-augment-batch-dl)](https://napari-hub.org/plugins/napari-easy-augment-batch-dl)

A plugin to perform unet based deep learning with a small number of labels and augmentation

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

and review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

To install latest development version :

    pip install git+https://github.com/bnorthan/napari-easy-augment-batch-dl.git

You will also need to install the latest development version of tnia-python:

    pip install git+https://github.com/True-North-Intelligent-Algorithms/tnia-python.git

You will need to install napari and for augmentation you will need albumentations library.  Also explicitly install numpy 1.26.  (We have not tested with numpy 2.0 so it is a good idea to explicitly install numpy 1.26 to avoid another dependency installing numpy 2.x)

```
    pip install numpy==1.26
    pip install napari[all]
    pip install albumentations
    pip install matplotlib
```

You will also need one or more of stardist, cellpose, segment-everything or Yolo

### Stardist

#### Windows

```
    conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
    pip install "tensorflow<2.11"
    pip install stardist==0.8.5
    pip install gputools
    pip install edt
```

#### Linux

```
    pip install tensorflow[and-cuda]
    pip install stardist
    pip install gputools
    pip install edt
```

### Pytorch (for unet segmentation)

```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install pytorch-lightning
    pip install monai
    pip install scipy
    pip install tifffile
```

### Cellpose

```
    pip install cellpose
```

### SAM (Segment Anything)

```
    pip install segment-everything
```

###

(more details to come on installing dependencies)


(Coming soon) You can install `napari-easy-augment-batch-dl` via [pip]:

    pip install napari-easy-augment-batch-dl



## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-easy-augment-batch-dl" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/bnorthan/napari-easy-augment-batch-dl/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
