
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject, ImageType
from enum import Enum
import os


def test_deep_learning_project():
    print('cwd is',os.getcwd())

    test_data_path = r'./test_data'
    parent_path = os.path.join(test_data_path, ImageType.GRAY_SCALE_2D.value)

    # confirm parent path exists
    assert os.path.exists(parent_path), f"Parent path {parent_path} does not exist"
    
    deep_learning_project = DeepLearningProject(parent_path, 2)

    print('deep_learning_project', type(deep_learning_project))
    print('number classes is', deep_learning_project.num_classes)

    num_images = len(deep_learning_project.image_file_list)

    assert num_images == 2, f"Expected 2 images, found {num_images}"

    print('number of images is', len(deep_learning_project.image_file_list))
    print('image names are', deep_learning_project.image_file_list)
    
    for i in range(len(deep_learning_project.image_list)):
        image = deep_learning_project.image_list[i]
        print('image', i, type(image), image.shape)