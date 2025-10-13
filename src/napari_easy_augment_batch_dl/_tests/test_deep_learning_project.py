
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject, ImageType
import os
import shutil 
import pytest
import logging
from pathlib import Path
import json
from napari_easy_augment_batch_dl.deep_learning_project import x1y1x2y2_to_tltrblbr

# Set up logger for this test module
logger = logging.getLogger(__name__)

# Available image types that have test data
AVAILABLE_IMAGE_TYPES = [
    ImageType.GRAY_SCALE_2D,
    ImageType.RGB_2D,
    ImageType.GRAY_SCALE_3D,
]

@pytest.fixture(params=AVAILABLE_IMAGE_TYPES, ids=lambda x: x.value)
def deep_learning_project(request):
    """Parametrized fixture that creates a DeepLearningProject for each image type."""
    logger.info(f'Setting up test for {request.param.value}')
    logger.debug(f'Current working directory: {os.getcwd()}')

    test_data_path = r'./test_data'
    image_type = request.param
    
    parent_path = os.path.join(test_data_path, image_type.value)
    parent_path_temp = os.path.join(test_data_path, 'temp', image_type.value)
    
    # Create the temporary directory
    os.makedirs(parent_path_temp, exist_ok=True)
    
    # Confirm parent path exists
    if not os.path.exists(parent_path):
        logger.warning(f"Test data for {image_type.value} not found at {parent_path}")
        pytest.skip(f"Test data for {image_type.value} not found at {parent_path}")
    
    # Copy the parent_path to the temporary directory
    shutil.copytree(parent_path, parent_path_temp, dirs_exist_ok=True)
    logger.debug(f"Copied test data from {parent_path} to {parent_path_temp}")
    
    # Create the deep learning project
    project = DeepLearningProject(parent_path_temp, 2)
    logger.info(f"Created DeepLearningProject with {len(project.image_file_list)} images")
    
    yield project
    
    # Cleanup after test
    shutil.rmtree(parent_path_temp, ignore_errors=True)
    logger.debug(f"Cleaned up temporary directory: {parent_path_temp}")

@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_directory():
    """Session-scoped fixture to ensure temp directory is cleaned up."""
    yield
    # Final cleanup of the entire temp directory
    temp_dir = './test_data/temp'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.info("Final cleanup of temp directory completed")

def test_deep_learning_project(deep_learning_project):
    """Test basic functionality of DeepLearningProject for all image types."""
    logger.info(f'Testing DeepLearningProject: {type(deep_learning_project)}')
    logger.info(f'Number of classes: {deep_learning_project.num_classes}')

    num_images = len(deep_learning_project.image_file_list)
    logger.info(f'Number of images: {num_images}')

    assert num_images >= 1, f"Expected at least 1 image, found {num_images}"

    logger.debug(f'Image file list: {deep_learning_project.image_file_list}')

    for i, image in enumerate(deep_learning_project.image_list):
        logger.debug(f'Image {i}: type={type(image)}, shape={image.shape}')

def test_augmentation_project(deep_learning_project):
    """Test augmentation functionality for all image types."""
    logger.info("Starting augmentation tests")
    perform_augmentation(deep_learning_project)
    perform_augmentation(deep_learning_project, do_color_jitter=True)
    logger.info("Completed augmentation tests")

def perform_augmentation(deep_learning_project, do_horizontal_flip=True, do_vertical_flip=True, do_random_rotate90=True, do_random_sized_crop=True, 
        do_random_brightness_contrast=True, do_random_gamma=False, do_color_jitter=False, do_elastic_transform=False):
    """Perform augmentation test with bounding box data."""
    
    # Check if we have any images
    if not deep_learning_project.image_file_list:
        logger.warning("No images available for augmentation test")
        pytest.skip("No images available for augmentation test")
    
    # Check if we have label paths
    if not deep_learning_project.image_label_paths:
        logger.warning("No label paths available for augmentation test")
        pytest.skip("No label paths available for augmentation test")
    
    json_names = list(Path(deep_learning_project.image_label_paths[0]).glob('*.json'))
    
    if not json_names:
        logger.warning("No JSON label files found for augmentation test")
        pytest.skip("No JSON label files found for augmentation test")
    
    image_name = deep_learning_project.image_file_list[0]
    image_base_name = image_name.name.split('.')[0]
    
    json_names_ = [x for x in json_names if image_base_name in x.name]

    logger.debug(f'Image base name: {image_base_name}')
    logger.debug(f'All JSON files: {json_names}')
    logger.debug(f'Matching JSON files: {json_names_}')
    
    if not json_names_:
        logger.warning(f"No JSON files found matching image {image_base_name}")
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
            logger.debug(f"Loaded bounding box from {json_name}: {bbox}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.error(f"Invalid JSON format in {json_name}: {e}")
        pytest.skip(f"Invalid JSON format in {json_name}: {e}")

    logger.info(f"Performing augmentation with {len(boxes)} bounding boxes")
    deep_learning_project.perform_augmentation(boxes, num_patches=100, patch_size=256, do_horizontal_flip=do_horizontal_flip, do_vertical_flip=do_vertical_flip,
                                               do_random_rotate90=do_random_rotate90, do_random_sized_crop=do_random_sized_crop, do_random_brightness_contrast=do_random_brightness_contrast,
                                               do_random_gamma=do_random_gamma, do_color_jitter=do_color_jitter, do_elastic_transform=do_elastic_transform)
    logger.info("Augmentation completed successfully")


