import napari
import os
from napari_easy_augment_batch_dl import easy_augment_batch_dl
from napari_easy_augment_batch_dl.deep_learning_project import DLModel

viewer = napari.Viewer()

batch_dl = easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer)

viewer.window.add_dock_widget(
    batch_dl
)

#test_dataset = "Roots" # "DAPI Cellpse"
test_dataset = "None"

#parent_path = os.path.join(r'D:\images\tnia-python-images\imagesc\2024_08_21_spheroids\yolo_omero_dataset\subset1')

if test_dataset == "Roots":
    parent_path =r'D:\images\tnia-python-images\imagesc\2024_08_15_plant_roots'
elif test_dataset == "DAPI Cellpose":
    parent_path =r'D:\images\tnia-python-images\imagesc\2024_08_19_Dapi_CJ'
elif test_dataset == "Bees":
    parent_path =r'C:\Users\bnort\work\ImageJ2022\tnia\gpu-image-analysis-deep-learning\data\ab_bees'
elif test_dataset == "Ladybugs":
    parent_path = r"C:\Users\bnort\work\ImageJ2022\tnia\gpu-image-analysis-deep-learning\data\aa_lady_bugs_fresh"
    parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_03_cellpose_ladybugs'
elif test_dataset == "Vesicles":
    parent_path =r'D:\images\tnia-python-images\tg\2024-09-26-vesicles'
elif test_dataset == "multi_uclear":
    parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_07_cellpose_multi_nuclear'

parent_path = r'D:\images\tnia-python-images\imagesc\2024_10_09_spindle_shaped_cells'

batch_dl.load_image_directory(parent_path)

# load the model that goes with the test
try:

    if test_dataset == "Roots":
        model_path = os.path.join(parent_path, r'models')
        model_name =  'model1' # None
        batch_dl.deep_learning_project.set_pretrained_model(os.path.join(model_path, model_name), DLModel.UNET)
        batch_dl.network_architecture_drop_down.setCurrentText(DLModel.UNET)
    elif test_dataset == "DAPI Cellpose":
        model_path = os.path.join(parent_path, r'models\models\cellpose_testing')
        batch_dl.deep_learning_project.set_pretrained_model(model_path, DLModel.CELLPOSE)
        batch_dl.network_architecture_drop_down.setCurrentText(DLModel.CELLPOSE)
    elif test_dataset == "Bees":
        model_path = os.path.join(parent_path, r'models')
        model_name = os.path.join(model_path, 'beetestmodel')
        #batch_dl.deep_learning_project.set_pretrained_model(model_name, DLModel.STARDIST)
        batch_dl.set_pretrained_model(model_name, "Stardist Model")
        batch_dl.network_architecture_drop_down.setCurrentText(DLModel.STARDIST)
except Exception as e:
    print('Exception occurred when loading model ', e)

napari.run()


model_path = os.path.join(parent_path, 'models')

batch_dl.load_image_directory(parent_path)

