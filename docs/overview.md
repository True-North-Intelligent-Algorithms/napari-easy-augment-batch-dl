# Overview

Napari-Easy-Augment-Batch-DL consists of a panel that interacts with several different Napari layers.  The below screenshot shows Napari-Easy-Augment-Batch-DL open in Napari.  

![Main Screen](images/main_screen_shot.png)
*Main screen showing various layers and annotations.*
## Panel Groups

 1. **Load and Label**: Load images then use the layers to label.

 2. **Augment**:  Apply augmentations to labels and save patches

 3. **Train and Predict**: Use patches to train models, then use model to generate predictions. 

## Napari-Easy-Augment-Batch Layers

 1. **Images:** The image set you are working with.

 2. **Labels:** Manually drawn labels.

 3. **Predictions:** Predictions from the trained model.

 4. **Label Box:** Indicates which regions should be used for training. Manually drawn labels outside of a label box will not be used for training. In the figure above, the blue box is a label box.

 5. **ml_labels (experimental):** Labels used for live ML training. Experiment with live machine learning training using these labels.

 6. **Object Box (experimental):** Annotate individual objects with a bounding box. This type of annotation is used for training YOLO and other object detection methods.

 7. **Predicted Object Box (experimental):** Predicted object box from YOLO or other object detection frameworks.
