
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject, ImageType
import os
import shutil 
import pytest
from pathlib import Path
import json
from napari_easy_augment_batch_dl.deep_learning_project import x1y1x2y2_to_tltrblbr

# Available image types that have test data
AVAILABLE_IMAGE_TYPES = [
    ImageType.GRAY_SCALE_2D,
    ImageType.RGB_2D,
    ImageType.GRAY_SCALE_3D,
]

@pytest.fixture(params=AVAILABLE_IMAGE_TYPES, ids=lambda x: x.value)
def deep_learning_project(request):
    """Parametrized fixture that creates a DeepLearningProject for each image type."""
    print(f'Setting up test for {request.param.value}')
    print('cwd is', os.getcwd())

    test_data_path = r'./test_data'
    image_type = request.param
    
    parent_path = os.path.join(test_data_path, image_type.value)
    parent_path_temp = os.path.join(test_data_path, 'temp', image_type.value)
    
    # Create the temporary directory
    os.makedirs(parent_path_temp, exist_ok=True)
    
    # Confirm parent path exists
    if not os.path.exists(parent_path):
        pytest.skip(f"Test data for {image_type.value} not found at {parent_path}")
    
    # Copy the parent_path to the temporary directory
    shutil.copytree(parent_path, parent_path_temp, dirs_exist_ok=True)
    
    # Create the deep learning project
    project = DeepLearningProject(parent_path_temp, 2)
    
    yield project
    
    # Cleanup after test
    shutil.rmtree(parent_path_temp, ignore_errors=True)

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_directory():
    """Session-scoped fixture to ensure temp directory is cleaned up."""
    yield
    # Final cleanup of the entire temp directory
    temp_dir = './test_data/temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_deep_learning_project(deep_learning_project):
    """Test basic functionality of DeepLearningProject for all image types."""
    print('deep_learning_project', type(deep_learning_project))
    print('number classes is', deep_learning_project.num_classes)

    num_images = len(deep_learning_project.image_file_list)

    assert num_images >= 1, f"Expected at least 1 image, found {num_images}"

    print('number of images is', len(deep_learning_project.image_file_list))
    print('image names are', deep_learning_project.image_file_list)

    for i in range(len(deep_learning_project.image_list)):
        image = deep_learning_project.image_list[i]
        print('image', i, type(image), image.shape)

def test_augmentation_project(deep_learning_project):
    """Test augmentation functionality for all image types."""
    perform_augmentation(deep_learning_project)
    perform_augmentation(deep_learning_project, do_color_jitter=True)

def perform_augmentation(deep_learning_project, do_horizontal_flip=True, do_vertical_flip=True, do_random_rotate90=True, do_random_sized_crop=True, 
        do_random_brightness_contrast=True, do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):
    """Perform augmentation test with bounding box data."""
    
    # Check if we have any images
    if not deep_learning_project.image_file_list:
        pytest.skip("No images available for augmentation test")
    
    # Check if we have label paths
    if not deep_learning_project.image_label_paths:
        pytest.skip("No label paths available for augmentation test")
    
    json_names = list(Path(deep_learning_project.image_label_paths[0]).glob('*.json'))
    
    if not json_names:
        pytest.skip("No JSON label files found for augmentation test")
    
    image_name = deep_learning_project.image_file_list[0]
    image_base_name = image_name.name.split('.')[0]
    
    json_names_ = [x for x in json_names if image_base_name in x.name]

    print('image_base_name', image_base_name)
    print('json_names', json_names)
    print('filtered json_names', json_names_)
    
    if not json_names_:
        pytest.skip(f"No JSON files found matching image {image_base_name}")
    
    json_name = json_names_[0]

    boxes = []
    try:
        with open(json_name, 'r') as f:
            json_ = json.load(f)
            
            x1 = json_['bbox'][0]
            y1 = json_['bbox'][1]
            x2 = json_['bbox'][2]
            y2 = json_['bbox'][3]

            bbox = x1y1x2y2_to_tltrblbr(x1, y1, x2, y2, 1)
            boxes.append(bbox)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        pytest.skip(f"Invalid JSON format in {json_name}: {e}")

    deep_learning_project.perform_augmentation(boxes, num_patches=100, patch_size=256, do_horizontal_flip=do_horizontal_flip, do_vertical_flip=do_vertical_flip,
                                               do_random_rotate90=do_random_rotate90, do_random_sized_crop=do_random_sized_crop, do_random_brightness_contrast=do_random_brightness_contrast,
                                               do_random_gamma=do_random_gamma, do_color_jitter=do_color_jitter, do_elastic_transform=do_elastic_transform)


