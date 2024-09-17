
from email.mime import image
from qtpy.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QPushButton, QFileDialog, QMessageBox, QInputDialog, QTextBrowser, QProgressBar, QCheckBox, QComboBox, QSpinBox, QHBoxLayout, QLabel, QLineEdit, QStackedWidget, QGridLayout
from PyQt5.QtCore import QThread
from pathlib import Path
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject
from napari_easy_augment_batch_dl.widgets import LabeledSpinner
from napari_easy_augment_batch_dl.deep_learning_project import DLModel
import numpy as np
import pandas as pd
import os
from napari_easy_augment_batch_dl.utility import pad_to_largest
from tnia.gui.threads.pyqt5_worker_thread import PyQt5WorkerThread

class NapariEasyAugmentBatchDL(QWidget):

    def __init__(self, napari_viewer, parent=None, label_only = False):
        super().__init__()

        self.viewer = napari_viewer

        self.deep_learning_project = None
        self.model = None

        self.worker = None

        self.counter = 0

        self.init_ui(label_only)

    def init_ui(self, label_only = False):

        self.label_only = label_only

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

        # current file name label
        self.current_file_name_label = QLabel("Current file: None")
        self.label_layout.addWidget(self.current_file_name_label)

        layout.addWidget(self.label_parameters_group)

        # add augment parameters group
        self.augment_parameters_group = QGroupBox("2. Augment images")
        self.augment_layout = QGridLayout()
        self.augment_parameters_group.setLayout(self.augment_layout)

        # add horizontal flip check box
        self.horizontal_flip_check_box = QCheckBox("Horizontal Flip")
        self.horizontal_flip_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(
            self.horizontal_flip_check_box, 0, 0
        )

        # add vertical flip check box
        self.vertical_flip_check_box = QCheckBox("Vertical Flip")
        self.vertical_flip_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(
            self.vertical_flip_check_box, 0, 1
        )

        # add rotate check box
        self.random_rotate_check_box = QCheckBox("Random Rotate")
        self.random_rotate_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(
            self.random_rotate_check_box, 1, 0
        )

        # add random resize check box
        self.random_resize_check_box = QCheckBox("Random Resize")
        self.random_resize_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(
            self.random_resize_check_box, 1, 1
        )

        # add random brightness contrast check box
        self.random_brightness_contrast_check_box = QCheckBox(
            "Random Brightness/Contrast"
        )
        self.random_brightness_contrast_check_box.setChecked(True)
        self.augment_parameters_group.layout().addWidget(
            self.random_brightness_contrast_check_box, 2, 0
        )

        # add random gamma check box
        self.random_gamma_check_box = QCheckBox("Random Gamma")
        self.random_gamma_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(
            self.random_gamma_check_box, 2, 1
        )

        # add random adjust color check box
        self.random_adjust_color_check_box = QCheckBox("Random Adjust Color")
        self.random_adjust_color_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(
            self.random_adjust_color_check_box, 3, 0
        )

        # add elastic deformation check box
        self.elastic_deformation_check_box = QCheckBox("Elastic Deformation")
        self.elastic_deformation_check_box.setChecked(False)
        self.augment_parameters_group.layout().addWidget(
            self.elastic_deformation_check_box, 3, 1
        )

        # add number patches spin box and label
        num_patches_layout = QHBoxLayout()
        num_patches_label = QLabel("Patches per ROI:")
        self.number_patches_spin_box = QSpinBox()
        self.number_patches_spin_box.setRange(1, 1000)
        self.number_patches_spin_box.setValue(100)
        num_patches_layout.addWidget(num_patches_label)
        num_patches_layout.addWidget(self.number_patches_spin_box)
        self.augment_parameters_group.layout().addLayout(num_patches_layout, 4, 0)

        # add patch size spin box and label
        patch_size_layout = QHBoxLayout()
        patch_size_label = QLabel("Patch size:")
        self.patch_size_spin_box = QSpinBox()
        self.patch_size_spin_box.setRange(1, 1000)
        self.patch_size_spin_box.setValue(256)
        patch_size_layout.addWidget(patch_size_label)
        patch_size_layout.addWidget(self.patch_size_spin_box)
        self.augment_parameters_group.layout().addLayout(patch_size_layout, 4, 1)
        
        # add augment current button
        self.augment_current_button = QPushButton("Augment current image")
        self.augment_current_button.clicked.connect(self.augment_current)
        self.augment_parameters_group.layout().addWidget(self.augment_current_button, 5, 0)

        # add perform augmentation button
        self.perform_augmentation_button = QPushButton("Augment all images")
        self.perform_augmentation_button.clicked.connect(self.augment_all)
        self.augment_parameters_group.layout().addWidget(self.perform_augmentation_button, 5, 1)

        # add delete augmentations button
        self.delete_augmentations_button = QPushButton("Delete augmentations")
        self.delete_augmentations_button.clicked.connect(self.delete_augmentations)
        self.augment_parameters_group.layout().addWidget(self.delete_augmentations_button, 6,0)


        # add train network group
        self.train_predict_group = QGroupBox("3. Train/Predict")
        self.train_predict_layout = QVBoxLayout()
        
        # add network architecture drop down
        self.network_architecture_drop_down = QComboBox()
        self.network_architecture_drop_down.addItem(DLModel.STARDIST)
        self.network_architecture_drop_down.addItem(DLModel.UNET)
        self.network_architecture_drop_down.addItem(DLModel.CELLPOSE)
        self.network_architecture_drop_down.addItem(DLModel.MOBILE_SAM2)
        self.network_architecture_drop_down.addItem(DLModel.YOLO_SAM)

        self.train_predict_layout.addWidget(self.network_architecture_drop_down)
        
        def on_index_changed(index):
            self.stacked_model_params_layout.setCurrentIndex(index)

        self.network_architecture_drop_down.currentIndexChanged.connect(on_index_changed)

        self.stacked_model_params_layout = QStackedWidget()
        
        self.pytorch_semantic_params_layout = QVBoxLayout()
        self.widgetGroup2 = QWidget()
        self.pytorch_semantic_threshold = LabeledSpinner("Threshold", -10, 10, 0, None, is_double=True, step=0.1)
        self.pytorch_semantic_params_layout.addWidget(self.pytorch_semantic_threshold)
        self.widgetGroup2.setLayout(self.pytorch_semantic_params_layout)

        self.stardist_params_layout = QVBoxLayout()
        self.widgetGroup1 = QWidget()
        self.prob_thresh = LabeledSpinner("Prob. Threshold", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.nms_thresh = LabeledSpinner("NMS Threshold", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.scale = LabeledSpinner("Scale", 0, 1, 0.5, None, is_double=True, step=0.01)
        self.stardist_params_layout.addWidget(self.prob_thresh)
        self.stardist_params_layout.addWidget(self.nms_thresh)
        self.stardist_params_layout.addWidget(self.scale)
        self.widgetGroup1.setLayout(self.stardist_params_layout)

        self.cellpose_params_layout = QVBoxLayout()
        self.widgetGroup3 = QWidget()
        self.cell_diameter_spinner = LabeledSpinner("Cell Diameter", 0, 100, 30, None, is_double=False, step=1)
        self.flow_threshold_spinner = LabeledSpinner("Flow Threshold", 0, 100, 0.4, None, is_double=True, step=0.01)
        self.cellpose_prob_thresh_spinner = LabeledSpinner("Prob. Threshold", 0, 1, 0.0, None, is_double=True, step=0.01)
        self.cellpose_params_layout.addWidget(self.cell_diameter_spinner)
        self.cellpose_params_layout.addWidget(self.flow_threshold_spinner)
        self.cellpose_params_layout.addWidget(self.cellpose_prob_thresh_spinner)  
        self.widgetGroup3.setLayout(self.cellpose_params_layout)
        
        self.mobile_sam_params_layout = QVBoxLayout()
        self.widgetGroup4 = QWidget()
        self.imagesz_ms = LabeledSpinner("Image Size", 0, 10000, 512, None, is_double=False, step=1)
        self.mobile_sam_params_layout.addWidget(self.imagesz_ms)
        self.widgetGroup4.setLayout(self.mobile_sam_params_layout)

        self.yolo_sam_params_layout = QVBoxLayout()
        self.widgetGroup5 = QWidget()
        self.imagesz_ys = LabeledSpinner("Image Size", 0, 10000, 512, None, is_double=False, step=1)
        self.yolo_class_label = QLabel("Class")
        self.yolo_class_drop_down = QComboBox()
        self.yolo_class_layout = QHBoxLayout()
        self.yolo_class_layout.addWidget(self.yolo_class_label)
        self.yolo_class_layout.addWidget(self.yolo_class_drop_down)

        # handle dropdown changed 
        def on_yolo_class_index_changed(index):
            if self.object_boxes_layer is not None:
                index = self.yolo_class_drop_down.currentIndex()
                self.object_boxes_layer.feature_defaults['class'] = index

        self.yolo_class_drop_down.currentIndexChanged.connect(on_yolo_class_index_changed) 

        self.yolo_sam_params_layout.addLayout(self.yolo_class_layout)
        self.yolo_sam_params_layout.addWidget(self.imagesz_ys)
        self.widgetGroup5.setLayout(self.yolo_sam_params_layout)

        self.stacked_model_params_layout.addWidget(self.widgetGroup1)
        self.stacked_model_params_layout.addWidget(self.widgetGroup2)
        self.stacked_model_params_layout.addWidget(self.widgetGroup3)
        self.stacked_model_params_layout.addWidget(self.widgetGroup4)
        self.stacked_model_params_layout.addWidget(self.widgetGroup5)

        self.train_predict_layout.addWidget(self.stacked_model_params_layout)
        
        self.train_predict_group.setLayout(self.train_predict_layout)
        
        # add load pretrained model button 
        self.load_pretrained_model_button = QPushButton("Load pretrained model...")
        self.load_pretrained_model_button.clicked.connect(self.load_pretrained_model)
        self.train_predict_layout.addWidget(self.load_pretrained_model_button)


        # add network name text box
        self.network_name_text_box = QLineEdit()
        self.network_name_text_box.setPlaceholderText("Enter network name")
        self.train_predict_layout.addWidget(self.network_name_text_box)

        # add number epochs spin box
        self.number_epochs_layout = QHBoxLayout()
        self.number_epochs_label = QLabel("Number of epochs:")
        self.number_epochs_spin_box = QSpinBox()
        self.number_epochs_spin_box.setRange(1, 1000)
        self.number_epochs_spin_box.setValue(100)
        self.number_epochs_layout.addWidget(self.number_epochs_label)
        self.number_epochs_layout.addWidget(self.number_epochs_spin_box)
        self.train_predict_layout.addLayout(self.number_epochs_layout)

        # add train network button
        self.train_network_button = QPushButton("Train network")
        self.train_network_button.clicked.connect(self.perform_training)
        self.train_predict_layout.addWidget(self.train_network_button)
        
        # add predict current image
        self.predict_current_image_button = QPushButton("Predict current image")
        self.predict_current_image_button.clicked.connect(self.predict_current_image)
        self.train_predict_layout.addWidget(self.predict_current_image_button)

        # add predict all images
        self.predict_all_images_button = QPushButton("Predict all images")
        self.predict_all_images_button.clicked.connect(self.predict_all_images)
        self.train_predict_layout.addWidget(self.predict_all_images_button)

        # add status log and progress
        self.textBrowser_log = QTextBrowser()
        self.progressBar = QProgressBar()

        if label_only == False:
            layout.addWidget(self.augment_parameters_group)
            layout.addWidget(self.train_predict_group)
            layout.addWidget(self.textBrowser_log)
            layout.addWidget(self.progressBar)
        else:
            self.predict_group = QGroupBox("Predict")
            self.predict_layout = QVBoxLayout()
            self.predict_layout.addWidget(self.predict_current_image_button)
            self.predict_layout.addWidget(self.predict_all_images_button)
            self.predict_group.setLayout(self.predict_layout)
            layout.addWidget(self.predict_group)
        self.setLayout(layout)

        def index_changed(event):
            index = self.viewer.dims.current_step[0]
            filename = self.deep_learning_project.files[index]
            self.current_file_name_label.setText(filename.name)
            
        
        
        self.viewer.dims.events.current_step.connect(index_changed)
    
    def update(self, message, progress=0):
        print('in the update')
        self.textBrowser_log.append(message)
        self.progressBar.setValue(progress)

    def update_thread(self, message, progress=0):
        if self.worker is not None:
            self.counter = self.counter + 1
            print('send signal to update ', self.counter)
            self.worker.progress.emit(message, progress)
        else:
            self.update(message, progress)

    def open_image_directory(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        image_path = QFileDialog.getExistingDirectory(
            self,
            "Select image directory",
            "",
            options=options,
        )

        self.load_image_directory(image_path)
    
    def load_image_directory(self, image_path):
        image_path = Path(image_path)

        files = list(image_path.glob('*.jpg'))
        files = files+list(image_path.glob('*.jpeg'))
        files = files+list(image_path.glob('*.tif'))
        files = files+list(image_path.glob('*.tiff'))
        files = files+list(image_path.glob('*.png'))

        if len(files) == 0:
            QMessageBox.information(self, "Error", "No images found in the selected directory. Please select a directory with images.")
            return
        
        # check if json exists
        
        if (image_path / 'info.json').exists():
            # pre-existing project num classes will be read from json
            num_classes = -1
            pass
        # else get num_classes
        else:
            num_classes, ok = QInputDialog.getInt(self, "Number of Classes", "Enter the number of classes (less than 8):", 1, 1, 8)

        self.deep_learning_project = DeepLearningProject(image_path, num_classes)
        
        self.images = pad_to_largest(self.deep_learning_project.image_list) #np.array(self.image_list)

        self.viewer.add_image(self.images, name='images')

        self.labels = []
        self.predictions = []

        for c in range(self.deep_learning_project.num_classes):
            #temp = pad_to_largest(self.deep_learning_project.label_list[c])
            temp = pad_to_largest(self.deep_learning_project.annotation_list[c])
            self.viewer.add_labels(temp, name='labels_'+str(c))
            self.labels.append(self.viewer.layers['labels_'+str(c)])

            temp = pad_to_largest(self.deep_learning_project.prediction_list[c])
            self.viewer.add_labels(temp, name='predictions_'+str(c))
            self.predictions.append(self.viewer.layers['predictions_'+str(c)])

        self.boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Label box",
            face_color="transparent",
            edge_color="blue",
            edge_width=5,
        )

        def handle_new_roi(event):
                    
            if event.action == 'added':
                
                box = self.boxes_layer.data[-1]
                z = int(box[0,0])
                ystart = int(np.min(box[:,1]))
                yend = int(np.max(box[:,1]))
                xstart = int(np.min(box[:,2]))
                xend = int(np.max(box[:,2]))

                if yend - ystart < self.patch_size_spin_box.value():
                    yend = yend + self.patch_size_spin_box.value()
                    if yend > self.images.shape[1]:
                        yend = self.images.shape[1]-1
                        ystart = yend - self.patch_size_spin_box.value()

                    new_box = np.array([[z, ystart, xstart], [z, ystart, xend], [z, yend, xend], [z, yend, xstart]])
                    self.boxes_layer.data[-1] = new_box
                    self.boxes_layer.refresh()

                if xend - xstart < self.patch_size_spin_box.value():
                    xend = xend + self.patch_size_spin_box.value()
                    if xend > self.images.shape[2]:
                        xend = self.images.shape[2]-1
                        xstart = xend - self.patch_size_spin_box.value()

                    new_box = np.array([[z, ystart, xstart], [z, ystart, xend], [z, yend, xend], [z, yend, xstart]])
                    self.boxes_layer.data[-1] = new_box
                    self.boxes_layer.refresh()

                for c in range(self.deep_learning_project.num_classes):
                    prediction =  self.predictions[c].data[z, ystart:yend, xstart:xend]

                    if np.sum(prediction) > 0:

                        reply = QMessageBox.question(self, 'Overwrite', 'Overwrite existing labels with predictions?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                        if reply == QMessageBox.Yes:
                            # copy from prediction to label
                                self.labels[c].data[z, ystart:yend, xstart:xend] = self.predictions[c].data[z, ystart:yend, xstart:xend]
                                self.labels[c].refresh()

                                print(self.boxes_layer.data)

        
        self.object_boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Object box",
            face_color="transparent",
            edge_color="green",
            edge_width=5,
            #label_property = text_property,
            #labels = annotations
            text={'string': '{class}', 'size': 15, 'color': 'green'},
        )
        
        self.predicted_object_boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Predicted Object box",
            face_color="transparent",
            edge_color="green",
            edge_width=5,
            #label_property = text_property,
            #labels = annotations
            text={'string': '{class}', 'size': 15, 'color': 'green'},
        )

        if self.label_only == True:
            self.object_boxes_layer.visible = False
            self.predicted_object_boxes_layer.visible = False

        for c in range(self.deep_learning_project.num_classes):
            self.yolo_class_drop_down.addItem("Class "+str(c))

        self.object_boxes_layer.feature_defaults['class'] = self.yolo_class_drop_down.currentIndex()
        self.predicted_object_boxes_layer.feature_defaults['class'] = self.yolo_class_drop_down.currentIndex()

        if self.deep_learning_project.boxes is not None:
            self.boxes_layer.add(self.deep_learning_project.boxes)

        if self.deep_learning_project.object_boxes is not None:
            self.object_boxes_layer.add(self.deep_learning_project.object_boxes)

        if self.deep_learning_project.predicted_object_boxes is not None:
            self.predicted_object_boxes_layer.add(self.deep_learning_project.predicted_object_boxes)

        if self.deep_learning_project.features is not None:
            self.object_boxes_layer.features = self.deep_learning_project.features

        if self.deep_learning_project.predicted_features is not None:
            self.predicted_object_boxes_layer.features = self.deep_learning_project.predicted_features
        
        self.boxes_layer.events.data.connect(handle_new_roi)
    
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

            self.deep_learning_project.set_pretrained_model(file_, DLModel.UNET)
            
        elif self.network_architecture_drop_down.currentText() == DLModel.STARDIST:
            self.textBrowser_log.append("Loading StarDist model...")
            start_model_path = QFileDialog.getExistingDirectory(self, "Select Model Directory", options=options)
            # Assuming it's a StarDist model
            self.deep_learning_project.set_pretrained_model(start_model_path, DLModel.STARDIST)
        elif self.network_architecture_drop_down.currentText() == DLModel.CELLPOSE:
            file_, _ = QFileDialog.getOpenFileName(self, "Select Cellpose Model File", "", "All Files (*)", options=options)
            self.deep_learning_project.set_pretrained_model(file_, DLModel.CELLPOSE)

        elif self.network_architecture_drop_down.currentText() == DLModel.MOBILE_SAM2:
            raise NotImplementedError("Mobile SAM2 model is fixed and cannot be changed")
        elif self.network_architecture_drop_down.currentText() == DLModel.YOLO_SAM:
            self.textBrowser_log.append("Loading YOLO model...")
            start_model_path = QFileDialog.getExistingDirectory(self, "Select Model Directory", options=options)
            self.deep_learning_project.set_pretrained_model(start_model_path, DLModel.YOLO_SAM)

    def save_results(self):
        label_nps = []

        for label in self.labels:
            label_nps.append(label.data)

        object_boxes=self.object_boxes_layer.data
        
        self.deep_learning_project.save_project(self.viewer.layers['Label box'].data, label_nps)

        if len(object_boxes)>0:        
            object_classes = self.object_boxes_layer.features['class'].to_numpy()
        
            self.deep_learning_project.save_object_boxes(object_boxes, object_classes)

        #QMessageBox.information(self, "Save Results", "Results saved successfully.")

    def augment_current(self):
        self.textBrowser_log.append("Augmenting current image...")

        n = self.viewer.dims.current_step[0]

        # get current boxes at n
        boxes = self.viewer.layers['Label box'].data
        boxes = np.array(boxes)
        index_boxes = np.all(boxes[:,:,0]==n, axis=1)
        filtered_boxes = boxes[index_boxes]
        self.perform_augmentation(filtered_boxes)

    def augment_all(self):
        boxes = self.viewer.layers['Label box'].data
        self.perform_augmentation(boxes)

    def delete_augmentations(self):
        self.deep_learning_project.delete_augmentations()

    def perform_augmentation(self, boxes):
        num_patches_per_roi = self.number_patches_spin_box.value()
        patch_size = self.patch_size_spin_box.value()

        perform_horizontal_flip = self.horizontal_flip_check_box.isChecked()
        perform_vertical_flip = self.vertical_flip_check_box.isChecked()
        perform_random_rotate = self.random_rotate_check_box.isChecked()
        perform_random_resize = self.random_resize_check_box.isChecked()
        perform_random_brightness_contrast = self.random_brightness_contrast_check_box.isChecked()
        perform_random_gamma = self.random_gamma_check_box.isChecked()
        perform_random_adjust_color = self.random_adjust_color_check_box.isChecked()
        perform_elastic_deformation = self.elastic_deformation_check_box.isChecked()

        self.textBrowser_log.append("Performing augmentation...")
        objects=self.object_boxes_layer.data
        # if yolo
        if self.network_architecture_drop_down.currentText() == DLModel.YOLO_SAM:
            classes = self.object_boxes_layer.features['class'].to_numpy()
            self.deep_learning_project.perform_yolo_augmentation(boxes, objects, classes, num_patches_per_roi, patch_size, self.update,
                                                                 perform_horizontal_flip, perform_vertical_flip, perform_random_rotate, perform_random_resize, 
                                                                 perform_random_brightness_contrast, perform_random_gamma, perform_random_adjust_color)
        else:
            self.deep_learning_project.perform_augmentation(boxes, num_patches_per_roi, patch_size, self.update,
                                                                 perform_horizontal_flip, perform_vertical_flip, perform_random_rotate, perform_random_resize, 
                                                                 perform_random_brightness_contrast, perform_random_gamma, perform_random_adjust_color)

    def perform_training(self):
        self.textBrowser_log.append("Training network...")

        num_epochs = self.number_epochs_spin_box.value()

        # perform another round of augmentation before training
        # The user also has the option to perform augmentation before training as to customize the augmented data
        # (for example they can augment a single image multiple times potentially with different augmentations, weighting that image more)
        # So ideally the user would augment the data before training
        # However this will confuse some users, as if they don't augment, not all images will be used in training
        # Thus we do a final round of augmentation before training

        #self.augment_all()

        thread = True
        if thread:
            '''
            if hasattr(self, 'thread'):
                if self.thread.isRunning():

                    self.thread.quit()
                    self.thread.wait()
            '''            
            self.thread = QThread()
            model = self.deep_learning_project.get_model(self.network_architecture_drop_down.currentText())
            model.create_callback(self.update_thread)
            self.worker = PyQt5WorkerThread(self.deep_learning_project.perform_training, self.network_architecture_drop_down.currentText(), num_epochs, None)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.disable_gui)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(self.update)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.enable_gui)

            self.thread.start()
             
        else:
            self.deep_learning_project.perform_training(self.network_architecture_drop_down.currentText(), num_epochs, self.update)

    def predict_current_image(self):
        self.textBrowser_log.append("Predicting current image...")
        n = self.viewer.dims.current_step[0]
        model_text = self.network_architecture_drop_down.currentText()
        if model_text == DLModel.YOLO_SAM or model_text == DLModel.MOBILE_SAM2:
            
            if model_text == DLModel.YOLO_SAM:
                imagesz = self.imagesz_ys.spinner.value()
            else:
                imagesz = self.imagesz_ms.spinner.value()
            
            predictions, boxes = self.deep_learning_project.predict(n, model_text, self.update, imagesz=imagesz)
            
            #self.object_boxes_layer.add(boxes)
            #self.object_boxes_layer.refresh()

            self.predicted_object_boxes_layer.add(boxes)
            self.predicted_object_boxes_layer.refresh()

            self.predictions[0].data[n, :predictions.shape[0], :predictions.shape[1]]=predictions
            self.predictions[0].refresh() 

        else:

            if model_text == DLModel.CELLPOSE:
                self.deep_learning_project.set_cellpose_params(self.cell_diameter_spinner.spinner.value(), [0,1], self.flow_threshold_spinner.spinner.value(), self.cellpose_prob_thresh_spinner.spinner.value())
            elif model_text == DLModel.STARDIST:
                self.deep_learning_project.set_stardist_params(self.prob_thresh.spinner.value(), self.nms_thresh.spinner.value(), self.scale.spinner.value())
            elif model_text == DLModel.UNET:
                self.deep_learning_project.set_pytorch_semantic_params(self.pytorch_semantic_threshold.spinner.value())

            pred = self.deep_learning_project.predict(n, self.network_architecture_drop_down.currentText(), self.update)
                
            self.predictions[0].data[n, :pred.shape[0], :pred.shape[1]]=pred
            self.predictions[0].refresh()              
        pass
    
    def predict_all_images(self):
        
        self.textBrowser_log.append("Predicting all images...")
        
        model_text = self.network_architecture_drop_down.currentText()
        if model_text == DLModel.YOLO_SAM or model_text == DLModel.MOBILE_SAM2:

            self.deep_learning_project.predict_all(model_text, self.update)

            #self.object_boxes_layer.add(self.deep_learning_project.object_boxes)
            self.predicted_object_boxes_layer.add(self.deep_learning_project.predicted_object_boxes)

            predictions = pad_to_largest(self.deep_learning_project.prediction_list[0])
            self.predictions[0].data = predictions
        
        else:
            self.deep_learning_project.predict_all(model_text, self.update)          
            predictions = pad_to_largest(self.deep_learning_project.prediction_list[0])

            #self.viewer.add_labels(predictions, name='predictions')
            self.predictions[0].data = predictions

    def disable_gui(self):
        pass

    def enable_gui(self):
        pass

       