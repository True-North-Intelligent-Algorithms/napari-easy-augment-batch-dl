
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject, ImageType
import os
import shutil 
import pytest
from pathlib import Path
import json
from napari_easy_augment_batch_dl.deep_learning_project import x1y1x2y2_to_tltrblbr

@pytest.fixture(scope="module", autouse=True)
def setup_and_cleanup():
    print('cwd is',os.getcwd())

    test_data_path = r'./test_data'

    parent_path_gray_scale_2d = os.path.join(test_data_path, ImageType.GRAY_SCALE_2D.value)
    parent_path_gray_scale_2d_temp = os.path.join(test_data_path, 'temp', ImageType.GRAY_SCALE_2D.value)
    # Create the temporary directory
    os.makedirs(parent_path_gray_scale_2d_temp, exist_ok=True)
    # confirm parent path exists
    assert os.path.exists(parent_path_gray_scale_2d), f"Parent path {parent_path_gray_scale_2d} does not exist"
    # Copy the parent_path to the temporary directory
    shutil.copytree(parent_path_gray_scale_2d, parent_path_gray_scale_2d_temp, dirs_exist_ok=True)
    # Run the test on the temporary directory
    deep_learning_project_gray_scale_2d = DeepLearningProject(parent_path_gray_scale_2d_temp, 2)

    parent_path_rgb_2d = os.path.join(test_data_path, ImageType.RGB_2D.value)
    parent_path_rgb_2d_temp = os.path.join(test_data_path, 'temp', ImageType.RGB_2D.value)
    # Create the temporary directory
    os.makedirs(parent_path_rgb_2d_temp, exist_ok=True)
    # confirm parent path exists
    assert os.path.exists(parent_path_rgb_2d), f"Parent path {parent_path_rgb_2d} does not exist"
    # Copy the parent_path to the temporary directory
    shutil.copytree(parent_path_rgb_2d, parent_path_rgb_2d_temp, dirs_exist_ok=True)
    deep_learning_project_rgb_2d = DeepLearningProject(parent_path_rgb_2d_temp, 2)

    yield deep_learning_project_gray_scale_2d, deep_learning_project_rgb_2d

    # Cleanup after test
    shutil.rmtree(parent_path_gray_scale_2d_temp, ignore_errors=True)
    shutil.rmtree(parent_path_rgb_2d_temp, ignore_errors=True)

def test_deep_learning_project(setup_and_cleanup):
    deep_learning_project, _ = setup_and_cleanup
    print('deep_learning_project', type(deep_learning_project))
    print('number classes is', deep_learning_project.num_classes)

    num_images = len(deep_learning_project.image_file_list)

    assert num_images == 2, f"Expected 2 images, found {num_images}"

    print('number of images is', len(deep_learning_project.image_file_list))
    print('image names are', deep_learning_project.image_file_list)

    for i in range(len(deep_learning_project.image_list)):
        image = deep_learning_project.image_list[i]
        print('image', i, type(image), image.shape)

def test_augmentation_project(setup_and_cleanup):
    perform_augmentation(setup_and_cleanup[0])
    perform_augmentation(setup_and_cleanup[1], do_color_jitter=True)

def perform_augmentation(deep_learning_project, do_horizontal_flip=True, do_vertical_flip=True, do_random_rotate90=True, do_random_sized_crop=True, 
        do_random_brightness_contrast=True, do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):
    
    json_names = list(Path(deep_learning_project.image_label_paths[0]).glob('*.json'))
    image_name = deep_learning_project.image_file_list[0]
    image_base_name = image_name.name.split('.')[0]
    #image_base_name = Path(image_name).stem
    
    json_names_ = [x for x in json_names if image_base_name in x.name]

    print('image_base_name', image_base_name)
    print('json_names', json_names)
    print('filtered json_names', json_names_)
    
    json_name = json_names_[0]

    boxes = []
    with open(json_name, 'r') as f:
        json_ = json.load(f)
        #print(json_)
                                    
        x1= json_['bbox'][0]
        y1= json_['bbox'][1]
        x2= json_['bbox'][2]
        y2= json_['bbox'][3]

        bbox = x1y1x2y2_to_tltrblbr(x1, y1, x2, y2, 1)
        boxes.append(bbox)        

     
    deep_learning_project.perform_augmentation(boxes, num_patches = 100, patch_size=256, do_horizontal_flip=do_horizontal_flip, do_vertical_flip=do_vertical_flip,
                                               do_random_rotate90=do_random_rotate90, do_random_sized_crop=do_random_sized_crop, do_random_brightness_contrast=do_random_brightness_contrast,
                                               do_random_gamma=do_random_gamma, do_color_jitter=do_color_jitter, do_elastic_transform=do_elastic_transform)



'''
def test_deep_learning_project():
    print('cwd is',os.getcwd())

    test_data_path = r'./test_data'
    parent_path = os.path.join(test_data_path, ImageType.GRAY_SCALE_2D.value)
    parent_path_temp = os.path.join(test_data_path, 'temp', ImageType.GRAY_SCALE_2D.value)

    # Create the temporary directory
    os.makedirs(parent_path_temp, exist_ok=True)

    # confirm parent path exists
    assert os.path.exists(parent_path), f"Parent path {parent_path} does not exist"

    try:
        # Copy the parent_path to the temporary directory
        shutil.copytree(parent_path, parent_path_temp, dirs_exist_ok=True)

        # Run the test on the temporary directory
        deep_learning_project = DeepLearningProject(parent_path_temp, 2)

        print('deep_learning_project', type(deep_learning_project))
        print('number classes is', deep_learning_project.num_classes)

        num_images = len(deep_learning_project.image_file_list)

        assert num_images == 2, f"Expected 2 images, found {num_images}"

        print('number of images is', len(deep_learning_project.image_file_list))
        print('image names are', deep_learning_project.image_file_list)

        for i in range(len(deep_learning_project.image_list)):
            image = deep_learning_project.image_list[i]
            print('image', i, type(image), image.shape)
    finally:
        # Clean up the temporary directory
        #shutil.rmtree(os.path.join(test_data_path, 'temp'))
        pass
'''