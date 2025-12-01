# Load and Label

## Preparation

Prior to using this plugin, put the images you want to work with in a project directory as shown below.  

![Project Directory](images/project_dir_start.png)

## 📌 Load Panel  

After starting the plugin the first step is to load your images and assign labels.  

![Load Panel](images/load_panel.png)  

1️⃣ Click the **Open image directory...** button.  
2️⃣ Select the directory that contains your image files.  

## Drawing Labels

1️⃣ Select **Label box** layer and draw a label box that is as large or larger than the desired patch size.  
2️⃣ Select **labels** layer and Label objects within the label box.

![Label Box and Labels](images/label_box_and_labels.png)

## Sparse vs Dense Labeling

Some algorithms support **sparse labeling**, which requires less work than dense labeling.

### Dense Labeling
Label every object in the image. Any pixels not labeled are implicitly assumed to be background. All pixels essentially have a label (either explicitly labeled objects or implicit background). Works with all frameworks.

### Sparse Labeling
Label only some objects and some background regions. Pixels with value 0 are treated as **unlabeled** and masked out during training (not used). This is faster but not all frameworks support it.

![Sparse Train Validation Label](images/sparse_train_validation_label.png)

**Important Notes:**
- Value 0 = **unlabeled** (masked out during training)
- Label 1 = background (must label some background explicitly)
- Labels 2+ = different object classes (for instance segmentation) or pixel classes (for semantic segmentation).
- Internally, the framework may subtract 1 from labels (making unlabeled pixels -1, background 0, etc.)

**Framework Support:**
Not all frameworks support sparse labeling. Check if your chosen framework supports this feature by:
- Consulting the framework documentation
- Asking on [image.sc](https://forum.image.sc/)
- Posting on other public forums with details about your problem

Sparse labeling can significantly reduce annotation time, but make sure your framework supports it before relying on this approach.

## Save Results

Select ```Save Results``` periodically to save the labels you have drawn.  

![Load Panel](images/load_panel.png)  

After saving results folders should be generated for different types of deep learning artifacts.  

![Project Directory](images/project_dir_save.png)

Inspect the labels directory to verify labels you have drawn have been saved.  

![Labels Directory](images/labels_dir.png)
---

🔄 **Next:** [Set Validation Labels](set_validation_labels.md)
