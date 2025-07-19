# note have to import napari first and show the viewer
# otherwise on Linux I get an error if I show the viewer after
# importing Cellpose
import napari
viewer = napari.Viewer()

import os
from napari_easy_augment_batch_dl import easy_augment_batch_dl

# import the frameworks you want to use.  As part of the import the framework wil be registered
# if it calls the BaseFramework.register_framework() method
try:
    from napari_easy_augment_batch_dl.frameworks.pytorch_semantic_framework import PytorchSemanticFramework
except Exception as e:
    print('PytorchSemanticFramework not loaded', e)
try:
    from napari_easy_augment_batch_dl.frameworks.monai_unet_framework import MonaiUNetFramework
except Exception as e:
    print('MonaiUnetFramework not loaded', e)
try:
    from napari_easy_augment_batch_dl.frameworks.cellpose_instance_framework import CellPoseInstanceFramework
except:
    print('CellPoseInstanceFramework not loaded')
try:
    from napari_easy_augment_batch_dl.frameworks.stardist_instance_framework import StardistInstanceFramework
except:
    print('StardistInstanceFramework not loaded')
try:
    from napari_easy_augment_batch_dl.frameworks.micro_sam_instance_framework import MicroSamInstanceFramework
except:
    print('MicroSamInstanceFramework not loaded')

# create the napari-easy-augment-batch-dl widget, we pass import_all_frameworks=False because
# we already imported the frameworks we want to use
batch_dl = easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer, import_all_frameworks=False)

from qtpy.QtWidgets import QScrollArea, QSizePolicy

scroll = QScrollArea()
scroll.setWidgetResizable(True)
scroll.setWidget(batch_dl)
#scroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

viewer.window.add_dock_widget(
    scroll
)

# here we list directories for image sets we have been experimenting with
# need to point the parent_path to a directory with the 2D image set we want to work with

#test_dataset = "Roots" # "DAPI Cellpse"
test_dataset = "None"
#test_dataset = "particles"
#test_dataset = "grains_cellpose"
#test_dataset = "DAPI Cellpose"
#t itest_dataset = "Bees"
#test_dataset = "grains_semantic"
#test_dataset = "custom"

#parent_path = os.path.join(r'D:\images\tnia-python-images\imagesc\2024_08_21_spheroids\yolo_omero_dataset\subset1')

if test_dataset == "Roots":
    parent_path =r'D:\images\tnia-python-images\imagesc\2024_08_15_plant_roots'
elif test_dataset == "DAPI Cellpose":
    parent_path =r'D:\images\tnia-python-images\imagesc\2024_08_19_Dapi_CJ'
elif test_dataset == "Bees":
    parent_path =r'C:\Users\bnort\work\ImageJ2022\tnia\gpu-image-analysis-deep-learning\data\ab_bees'
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2024_08_27_bees_sparse/'
elif test_dataset == "Ladybugs":
    parent_path = r"C:\Users\bnort\work\ImageJ2022\tnia\gpu-image-analysis-deep-learning\data\aa_lady_bugs_fresh"
    parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_03_cellpose_ladybugs'
elif test_dataset == "Vesicles":
    parent_path =r'D:\images\tnia-python-images\tg\2024-09-26-vesicles'
elif test_dataset == "multi_uclear":
    parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_07_cellpose_multi_nuclear'
elif test_dataset == "spindle":
    parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_09_spindle_shaped_cells'
elif test_dataset == "thin_section":
    parent_path =r'D:\images\tnia-python-images\\imagesc\\2024_10_17_thin_section'
elif test_dataset == "ladybugs":
    data_path = r'C:\Users\bnort\work\ImageJ2022\tnia\notebooks-and-napari-widgets-for-dl\data'
    parent_path = os.path.join(data_path, 'ladybugs1')
elif test_dataset == "particles":
    parent_path = r'/home/bnorthan/besttestset/images/Semantic_Spar/'
elif test_dataset == "grains_cellpose":
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2024_12_19_sem_grain_size_revisit2'
elif test_dataset == "grains_semantic":
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2024_12_19_sem_grain_size_revisit/'
elif test_dataset == "particles2":
    parent_path = r'/home/bnorthan/besttestset/images/training14/'
else:
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_03_28_vessel_3D_lightsheet'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_03_31_cellpose_not_precise'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_04_14_sheep_follicles'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_04_17_wood_stave_vessels'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_04_14_sheep_follicles2'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_05_10_SOTA_Test_Set'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_03_19_vessel_3D_lightsheet'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_06_03_cellpose_training'
    #parent_path = r'D:\images\tnia-python-images\imagesc\2025_06_21_tough_cellpose_subset'
    #parent_path = r'D:\images\tnia-python-images\imagesc\2025_05_31_Black_Bean'
    #parent_path = r'D:\images\tnia-python-images\imagesc\2025_04_10_beans'
    #parent_path = r'D:\images\tnia-python-images\imagesc\2025_06_15_bubbles'
    #parent_path = r'D:\images\tnia-python-images\imagesc\2025_03_05_bugs'
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2024_12_19_sem_grain_size_revisit'
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2024_12_19_sem_grain_size_revisit2'
    parent_path = r'/home/bnorthan/images/tnia-python-images/imagesc/2025_06_26_new_vessels'
    parent_path = r'D:\images\tnia-python-images\imagesc\2025_07_08_new_vessels'
    parent_path = r'D:\images\tnia-python-images\imagesc/2024_12_19_sem_grain_size_revisit/'

model_path = os.path.join(parent_path, r'models')

batch_dl.load_image_directory(parent_path)

# load the model that goes with the test
try:

    if test_dataset == "Roots":
        model_name =  None #'model1' 
        batch_dl.network_architecture_drop_down.setCurrentText(PytorchSemanticFramework.descriptor)
    elif test_dataset == "DAPI Cellpose":
        model_name = os.path.join(parent_path, r'models\models\cellpose_testing')
        batch_dl.network_architecture_drop_down.setCurrentText(CellPoseInstanceFramework.descriptor)
        widget = batch_dl.deep_learning_widgets[CellPoseInstanceFramework.descriptor]
        widget.load_model_from_path(model_name)
    elif test_dataset == "Bees":
        model_name = os.path.join(model_path, 'model1')
        batch_dl.network_architecture_drop_down.setCurrentText(StardistInstanceFramework.descriptor)
        widget = batch_dl.dSeep_learning_widgets[StardistInstanceFramework.descriptor]
        widget.load_model_from_path(model_name)
    elif test_dataset == "particles":
        model_name = 'ok.pth'
        widget = batch_dl.deep_learning_widgets[PytorchSemanticFramework.descriptor]
        widget.load_model_from_path(os.path.join(model_path, model_name))
    elif test_dataset == "grains cellpose":
        model_name = "cellpose_20241219_090937"
        batch_dl.network_architecture_drop_down.setCurrentText(CellPoseInstanceFramework.descriptor)
        widget = batch_dl.deep_learning_widgets[CellPoseInstanceFramework.descriptor]
        widget.load_model_from_path(os.path.join(model_path, model_name))
    elif test_dataset == 'grains_semantic':
        model_name = 'semantic_20250130_200224.pth'
        batch_dl.network_architecture_drop_down.setCurrentText(PytorchSemanticFramework.descriptor)
        widget = batch_dl.deep_learning_widgets[PytorchSemanticFramework.descriptor]
        widget.load_model_from_path(os.path.join(model_path, model_name))
        
except Exception as e:
    print('Exception occurred when loading model ', e)

model_path = os.path.join(parent_path, 'models')

napari.run()