
from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPushButton, QFileDialog, QMessageBox, QInputDialog, QTextBrowser, QProgressBar, QCheckBox, QComboBox, QSpinBox, QHBoxLayout, QLabel, QLineEdit, QStackedWidget
from pathlib import Path
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject
from napari_easy_augment_batch_dl.widgets import LabeledSpinner
from napari_easy_augment_batch_dl.deep_learning_project import DLModel
import numpy as np
import pandas as pd
import os

class NapariEasyAugmentBatchDL(QWidget):

    def __init__(self, viewer, parent=None):
        super().__init__()

        self.viewer = viewer

        self.deep_learning_project = None
        self.model = None

        self.init_ui()

    def init_ui(self):

        self.setWindowTitle("Easy Augment Batch DL")
        layout = QVBoxLayout()
        layout.setSpacing(30)

        # add label parameters group
        self.label_parameters_group = QGroupBox("1. Draw labels")
        self.label_layout = QVBoxLayout()
        self.label_parameters_group.setLayout(self.label_layout)


        # add open results button
        self.open_image_directory_button = QPushButton("Open image directory...")
        self.open_image_directory_button.clicked.connect(self.open_image_directory)
        self.label_layout.addWidget(self.open_image_directory_button)

        # add save results button 
        self.save_results_button = QPushButton("Save results...")
        self.save_results_button.clicked.connect(self.save_results)
        self.label_layout.addWidget(self.save_results_button)
        
        layout.addWidget(self.label_parameters_group)

        # add augment parameters group
        self.augment_parameters_group = QGroupBox("2. Augment images")
        self.augment_layout = QVBoxLayout()
        self.augment_parameters_group.setLayout(self.augment_layout)

        # add horizontal flip check box
        self.horizontal_flip_check_box = QCheckBox("Horizontal Flip")
        self.horizontal_flip_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(self.horizontal_flip_check_box)

        # add vertical flip check box
        self.vertical_flip_check_box = QCheckBox("Vertical Flip")
        self.vertical_flip_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(self.vertical_flip_check_box)

        # add rotate check box
        self.random_rotate_check_box = QCheckBox("Random Rotate")
        self.random_rotate_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(self.random_rotate_check_box)

        # add random resize check box
        self.random_resize_check_box = QCheckBox("Random Resize")
        self.random_resize_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(self.random_resize_check_box)

        # add random brightness contrast check box
        self.random_brightness_contrast_check_box = QCheckBox("Random Brightness/Contrast")
        self.random_brightness_contrast_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(self.random_brightness_contrast_check_box)    

        # add random gamma check box
        self.random_gamma_check_box = QCheckBox("Random Gamma")
        self.random_gamma_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(self.random_gamma_check_box)

        # add random adjust color check box
        self.random_adjust_color_check_box = QCheckBox("Random Adjust Color")
        self.random_adjust_color_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(self.random_adjust_color_check_box)

        # add elastic deformation check box
        self.elastic_deformation_check_box = QCheckBox("Elastic Deformation")
        self.elastic_deformation_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(self.elastic_deformation_check_box)

        # add number patches spin box and label
        num_patches_layout = QHBoxLayout()
        num_patches_label = QLabel("Patches per ROI:")
        self.number_patches_spin_box = QSpinBox()
        self.number_patches_spin_box.setRange(1, 1000)
        self.number_patches_spin_box.setValue(100)
        num_patches_layout.addWidget(num_patches_label)
        num_patches_layout.addWidget(self.number_patches_spin_box)
        self.augment_parameters_group.layout().addLayout(num_patches_layout)

        # add patch size spin box and label
        patch_size_layout = QHBoxLayout()
        patch_size_label = QLabel("Patch size:")
        self.patch_size_spin_box = QSpinBox()
        self.patch_size_spin_box.setRange(1, 1000)
        self.patch_size_spin_box.setValue(256)
        patch_size_layout.addWidget(patch_size_label)
        patch_size_layout.addWidget(self.patch_size_spin_box)
        self.augment_parameters_group.layout().addLayout(patch_size_layout)

        # add perform augmentation button
        self.perform_augmentation_button = QPushButton("Perform Augmentation")
        self.perform_augmentation_button.clicked.connect(self.perform_augmentation)
        self.augment_parameters_group.layout().addWidget(self.perform_augmentation_button)

        layout.addWidget(self.augment_parameters_group)

        # add train network group
        self.train_network_group = QGroupBox("3. Train network")
        self.train_layout = QVBoxLayout()
        
        # add network architecture drop down
        self.network_architecture_drop_down = QComboBox()
        self.network_architecture_drop_down.addItem(DLModel.UNET)
        self.network_architecture_drop_down.addItem(DLModel.STARDIST)
        self.network_architecture_drop_down.addItem(DLModel.CELLPOSE)
        self.network_architecture_drop_down.addItem(DLModel.YOLO_SAM)

        self.train_layout.addWidget(self.network_architecture_drop_down)

        def on_index_changed(index):
            self.stacked_model_params_layout.setCurrentIndex(index)

        self.network_architecture_drop_down.currentIndexChanged.connect(on_index_changed)

        self.stacked_model_params_layout = QStackedWidget()
        
        self.pytorch_semantic_params_layout = QVBoxLayout()
        self.widgetGroup1 = QWidget()
        self.pytorch_semantic_threshold = LabeledSpinner("Threshold", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.pytorch_semantic_params_layout.addWidget(self.pytorch_semantic_threshold)
        self.widgetGroup1.setLayout(self.pytorch_semantic_params_layout)

        self.stardist_params_layout = QVBoxLayout()
        self.widgetGroup2 = QWidget()
        self.prob_thresh = LabeledSpinner("Prob. Threshold", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.nms_thresh = LabeledSpinner("NMS Threshold", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.stardist_params_layout.addWidget(self.prob_thresh)
        self.stardist_params_layout.addWidget(self.nms_thresh)
        self.widgetGroup2.setLayout(self.stardist_params_layout)

        self.cellpose_params_layout = QVBoxLayout()
        self.widgetGroup3 = QWidget()
        self.cell_diameter = LabeledSpinner("Cell Diameter", 0, 100, 30, None, is_double=False, step=1)
        self.cellpose_params_layout.addWidget(self.cell_diameter)
        self.widgetGroup3.setLayout(self.cellpose_params_layout)

        self.yolo_sam_params_layout = QVBoxLayout()
        self.widgetGroup4 = QWidget()
        self.imagesz = LabeledSpinner("Image Size", 0, 10000, 512, None, is_double=False, step=1)
        self.yolo_sam_params_layout.addWidget(self.imagesz)
        self.widgetGroup4.setLayout(self.yolo_sam_params_layout)

        self.stacked_model_params_layout.addWidget(self.widgetGroup1)
        self.stacked_model_params_layout.addWidget(self.widgetGroup2)
        self.stacked_model_params_layout.addWidget(self.widgetGroup3)
        self.stacked_model_params_layout.addWidget(self.widgetGroup4)

        self.train_layout.addWidget(self.stacked_model_params_layout)
        
        self.train_network_group.setLayout(self.train_layout)
        
        # add load pretrained model button 
        self.load_pretrained_model_button = QPushButton("Load pretrained model...")
        self.load_pretrained_model_button.clicked.connect(self.load_pretrained_model)
        self.train_layout.addWidget(self.load_pretrained_model_button)


        # add network name text box
        self.network_name_text_box = QLineEdit()
        self.network_name_text_box.setPlaceholderText("Enter network name")
        self.train_layout.addWidget(self.network_name_text_box)

        # add number epochs spin box
        self.number_epochs_layout = QHBoxLayout()
        self.number_epochs_label = QLabel("Number of epochs:")
        self.number_epochs_spin_box = QSpinBox()
        self.number_epochs_spin_box.setRange(1, 1000)
        self.number_epochs_spin_box.setValue(100)
        self.number_epochs_layout.addWidget(self.number_epochs_label)
        self.number_epochs_layout.addWidget(self.number_epochs_spin_box)
        self.train_layout.addLayout(self.number_epochs_layout)

        # add train network button
        self.train_network_button = QPushButton("Train network")
        self.train_network_button.clicked.connect(self.perform_training)
        self.train_layout.addWidget(self.train_network_button)
        
        layout.addWidget(self.train_network_group)

        # add predict group
        self.predict_group = QGroupBox("4. Predict")
        self.predict_layout = QVBoxLayout()
        self.predict_group.setLayout(self.predict_layout)

        # add predict current image
        self.predict_current_image_button = QPushButton("Predict current image")
        self.predict_current_image_button.clicked.connect(self.predict_current_image)
        self.predict_layout.addWidget(self.predict_current_image_button)

        # add predict all images
        self.predict_all_images_button = QPushButton("Predict all images")
        self.predict_all_images_button.clicked.connect(self.predict_all_images)
        self.predict_layout.addWidget(self.predict_all_images_button)

        layout.addWidget(self.predict_group)


        # add status log and progress
        self.textBrowser_log = QTextBrowser()
        self.progressBar = QProgressBar()

        layout.addWidget(self.textBrowser_log)
        layout.addWidget(self.progressBar)

        self.setLayout(layout)
    
    def update(self, message, progress=0):
        self.textBrowser_log.append(message)
        self.progressBar.setValue(progress)


    def open_image_directory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select image directory",
            "",
            options=options,
        )

        image_path = Path(directory)

        files = list(image_path.glob('*.jpg'))
        files = files+list(image_path.glob('*.jpeg'))
        files = files+list(image_path.glob('*.tif'))
        files = files+list(image_path.glob('*.png'))

        if len(files) == 0:
            QMessageBox.information(self, "Error", "No images found in the selected directory. Please select a directory with images.")
            return
        
        # check if json exists
        
        if (Path(directory) / 'info.json').exists():
            # pre-existing project num classes will be read from json
            num_classes = -1
            pass
        # else get num_classes
        else:
            num_classes, ok = QInputDialog.getInt(self, "Number of Classes", "Enter the number of classes (less than 8):", 1, 1, 8)

        self.deep_learning_project = DeepLearningProject(image_path, num_classes)

        self.viewer.add_image(self.deep_learning_project.images, name='images')

        self.labels = []

        for c in range(self.deep_learning_project.num_classes):
            self.viewer.add_labels(self.deep_learning_project.label_list[c], name='labels_'+str(c))
            self.labels.append(self.viewer.layers['labels_'+str(c)])

        boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Label box",
            face_color="transparent",
            edge_color="blue",
            edge_width=5,
        )

        self.object_boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Object box",
            face_color="transparent",
            edge_color="green",
            edge_width=5,
        )

        if self.deep_learning_project.boxes is not None:
            boxes_layer.add(self.deep_learning_project.boxes)

        if self.deep_learning_project.object_boxes is not None:
            self.object_boxes_layer.add(self.deep_learning_project.object_boxes)
    
    def load_pretrained_model(self):
         # Open a file dialog to select a file or directory
        options = QFileDialog.Options()

        if self.network_architecture_drop_down.currentText() == DLModel.UNET:
            file_, _ = QFileDialog.getOpenFileName(self, "Select Model File or Directory", "", "All Files (*)", options=options)

            if file_.lower().endswith('.pth'):
                #self.load_pytorch_model(file_or_directory)
                self.textBrowser_log.append("Loading Pytorch model...")
            else:
                self.textBrowser_log.append("Selected file is not a .pth file.")
        elif self.network_architecture_drop_down.currentText() == DLModel.STARDIST:
            self.textBrowser_log.append("Loading StarDist model...")
            start_model_path = QFileDialog.getExistingDirectory(self, "Select Model Directory", options=options)
            # Assuming it's a StarDist model
            self.deep_learning_project.set_pretrained_model(start_model_path)
        elif self.network_architecture_drop_down.currentText() == DLModel.CELLPOSE:
            pass

    def save_results(self):
        self.textBrowser_log.append("Saving results...")

        napari_path = self.deep_learning_project.napari_path
        
        if not napari_path.exists():
                os.makedirs(napari_path)
            
        for layer in self.viewer.layers:
            np.save(os.path.join(napari_path, layer.name+'.npy'), layer.data)

        self.deep_learning_project.save_project(self.viewer.layers['Label box'].data)

        #QMessageBox.information(self, "Save Results", "Results saved successfully.")

    def perform_augmentation(self):
        num_patches_per_roi = self.number_patches_spin_box.value()
        patch_size = self.patch_size_spin_box.value()
        self.textBrowser_log.append("Performing augmentation...")
        boxes=self.viewer.layers['Label box'].data
        objects=self.object_boxes_layer.data
        # if yolo
        if self.network_architecture_drop_down.currentText() == DLModel.YOLO_SAM:
            classes = self.object_boxes_layer.features['class'].to_numpy()
            self.deep_learning_project.perform_yolo_augmentation(boxes, objects, classes, num_patches_per_roi, patch_size, self.update)
        else:
            self.deep_learning_project.perform_augmentation(boxes, num_patches_per_roi, patch_size)

    def perform_training(self):
        self.textBrowser_log.append("Training network...")

        num_epochs = self.number_epochs_spin_box.value()

        self.deep_learning_project.perform_training(self.network_architecture_drop_down.currentText(), num_epochs, self.update)

    def predict_current_image(self):
        self.textBrowser_log.append("Predicting current image...")
        n = self.viewer.dims.current_step[0]
        if self.network_architecture_drop_down.currentText() == DLModel.YOLO_SAM:
            predictions, boxes = self.deep_learning_project.predict(n, DLModel.YOLO_SAM, self.update)
            self.object_boxes_layer.add(boxes)
            self.object_boxes_layer.refresh()
            self.labels[0].data[n, :predictions.shape[0], :predictions.shape[1]]=predictions
            self.labels[0].refresh() 
        else:
            pred = self.deep_learning_project.predict(n, self.network_architecture_drop_down.currentText(), self.update)
            self.labels[0].data[n, :pred.shape[0], :pred.shape[1]]=pred
            self.labels[0].refresh()              
        pass
    
    def predict_all_images(self):
        
        self.textBrowser_log.append("Predicting all images...")
        
        if self.network_architecture_drop_down.currentText() == DLModel.YOLO_SAM:

            predictions, boxes = self.deep_learning_project.predict_all(DLModel.YOLO_SAM, self.update)

            self.object_boxes_layer.add(boxes)
            predictions = self.deep_learning_project.pad_to_largest(predictions)
            self.labels[0].data = predictions
        else:
            predictions = self.deep_learning_project.predict_all(self.network_architecture_drop_down.currentText(), self.update)          
            predictions = self.deep_learning_project.pad_to_largest(predictions)

            #self.viewer.add_labels(predictions, name='predictions')
            self.labels[0].data = predictions




       