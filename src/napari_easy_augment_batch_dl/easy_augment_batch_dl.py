from qtpy.QtWidgets import (
    QDialog, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QGroupBox,
    QPushButton, QFileDialog, QMessageBox,
    QInputDialog, QTextBrowser, QProgressBar,
    QCheckBox, QComboBox, QSpinBox,
    QLabel, QStackedWidget, QSizePolicy,
)
from napari_easy_augment_batch_dl.widgets import LabeledSpinner, LabeledCombo
from PyQt5.QtCore import QThread
from pathlib import Path
from napari_easy_augment_batch_dl.deep_learning_project import DeepLearningProject
import numpy as np
from napari_easy_augment_batch_dl.utility import pad_to_largest, unpad_to_original
from tnia.gui.threads.pyqt5_worker_thread import PyQt5WorkerThread
from napari_easy_augment_batch_dl.deep_learning_widget import DeepLearningWidget
try:
    from segment_everything.detect_and_segment import segment_from_bbox, create_sam_model
except Exception as e:
    print(e)

from enum import Enum
import warnings

class EasyAugmentMode(Enum):
    ALL = "all"
    LABEL_ONLY = "label_only"
    DL_PIXEL_ONLY = "dl_pixel_only"

class NapariEasyAugmentBatchDL(QWidget):

    def __init__(self, napari_viewer, parent=None, label_only = False, import_all_frameworks = True, mode = None):
        super().__init__()

        if mode is not None:
            self.mode = mode
        else:
            # Backward compatibility
            if label_only:
                self.mode = EasyAugmentMode.LABEL_ONLY
            else:
                self.mode = EasyAugmentMode.ALL
            warnings.warn(
                "The 'label_only' parameter is deprecated. Please use the 'mode' parameter instead.",
                DeprecationWarning,
                stacklevel=2
            )

        if import_all_frameworks:
            self.import_all_frameworks()
            
        self.viewer = napari_viewer

        self.deep_learning_project = None
        self.model = None

        self.worker = None

        self.counter = 0

        self.init_ui()

    def import_all_frameworks(self):

        try:
            from napari_easy_augment_batch_dl.frameworks.stardist_instance_framework import StardistInstanceFramework
        except:
            StardistInstanceFramework = None
     
        try:
            from napari_easy_augment_batch_dl.frameworks.cellpose_instance_framework import CellPoseInstanceFramework
        except:
            print('CellPoseInstanceFramework not loaded')

        '''
        TODO: delete, monai unet framework replaces this
        try:
            from napari_easy_augment_batch_dl.frameworks.pytorch_semantic_framework import PytorchSemanticFramework
        except:
            print('PytorchSemanticFramework not loaded')
        '''
        
        try:
            from napari_easy_augment_batch_dl.frameworks.mobile_sam_framework import MobileSAMFramework
        except ImportError:
            MobileSAMFramework = None
        
        try:
            from napari_easy_augment_batch_dl.frameworks.yolo_sam_framework import YoloSAMFramework
        except ImportError:
            YoloSAMFramework = None

        try:
            from napari_easy_augment_batch_dl.frameworks.random_forest_framework import RandomForestFramework
        except ImportError:
            RandomForestFramework = None
        
        try:
            from napari_easy_augment_batch_dl.frameworks.monai_unet_framework import MonaiUNetFramework
        except Exception as e:
            print('MonaiUnetFramework not loaded', e)

        try:
            from napari_easy_augment_batch_dl.frameworks.micro_sam_instance_framework import MicroSamInstanceFramework
        except:
            print('MicroSamInstanceFramework not loaded')
    
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

        # add the save histogram checkbox
        self.save_histogram_check_box = QCheckBox("Save histograms")
        self.save_histogram_check_box.setChecked(False)
        self.label_layout.addWidget(self.save_histogram_check_box)

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
        self.patch_size_spin_box.setRange(1, 4096)
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

        # add advanced augmentation settings button
        self.augmentation_settings_button = QPushButton("Settings...")
        self.augmentation_settings_button.clicked.connect(self.augment_settings)
        self.augment_parameters_group.layout().addWidget(self.augmentation_settings_button, 6, 1)

        # add train network group
        self.train_predict_group = QGroupBox("3. Train/Predict")
        self.train_predict_layout = QVBoxLayout()
        
        # add network architecture drop down
        self.network_architecture_drop_down = QComboBox()
        self.train_predict_layout.addWidget(self.network_architecture_drop_down)
        
        def on_index_changed(index):
            self.stacked_model_params_layout.setCurrentIndex(index)

        self.network_architecture_drop_down.currentIndexChanged.connect(on_index_changed)

        self.stacked_model_params_layout = QStackedWidget()

        self.train_predict_layout.addWidget(self.stacked_model_params_layout)
        
        self.train_predict_group.setLayout(self.train_predict_layout)

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
        
        # add second save results button for bottom of panel 
        self.save_results_button2 = QPushButton("Save results...")
        self.save_results_button2.clicked.connect(self.save_results)
        self.train_predict_layout.addWidget(self.save_results_button2)

        # add status log and progress
        self.textBrowser_log = QTextBrowser()
        self.progressBar = QProgressBar()

        if str(self.mode) != str(EasyAugmentMode.LABEL_ONLY):
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
            filename = self.deep_learning_project.image_file_list[index]
            self.current_file_name_label.setText(filename.name)
        
        self.viewer.dims.events.current_step.connect(index_changed)

        # comment out for now TODO:  Revisit interactive segmentation
        # try to create a sam model which will be used for creating labels from bounding boxes
        try:
            from segment_everything.detect_and_segment import segment_from_bbox, create_sam_model
            self.helper_sam_model = create_sam_model("MobileSamV2", 'cuda')
        except Exception as e:
            print(e)
            self.helper_sam_model = None
    def update(self, message, progress=0):
        self.textBrowser_log.append(message)
        self.progressBar.setValue(int(progress))

    def update_thread(self, message, progress=0):
        if self.worker is not None:
            self.counter = self.counter + 1
            self.worker.progress.emit(message, progress)
        else:
            self.update(message, progress)

    def open_image_directory(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        
        parent_path = QFileDialog.getExistingDirectory(
            self,
            "Select image directory",
            "",
            options=options,
        )

        self.load_image_directory(parent_path)
    
    def load_image_directory(self, parent_path):
        self.parent_path = Path(parent_path)

        files = list(self.parent_path.glob('*.jpg'))
        files = files+list(self.parent_path.glob('*.jpeg'))
        files = files+list(self.parent_path.glob('*.tif'))
        files = files+list(self.parent_path.glob('*.tiff'))
        files = files+list(self.parent_path.glob('*.png'))

        if len(files) == 0:
            QMessageBox.information(self, "Error", "No images found in the selected directory. Please select a directory with images.")
            return
        
        # check if json exists
        
        # TODO: rethink whether user needs to define number of classes
        # related to one hot encoding vs unique label indexes for semantic segmentation
        if (self.parent_path / 'info.json').exists():
            # pre-existing project num classes will be read from json
            num_classes = -1
            pass
        # else get num_classes
        else:
            # for now comment out the 'enter number of classes' dialog and default to one
            # TODO: make an 'add classes' approach so as user is interacting can add classes in middle of project
            # num_classes, ok = QInputDialog.getInt(self, "Number of Classes", "Enter the number of classes (less than 8):", 1, 1, 8)
            num_classes = 1

        self.deep_learning_project = DeepLearningProject(self.parent_path, num_classes)
        self.deep_learning_widgets = {}

        # loop through the key and instance of all the deep learning frameworks
        for key, obj in self.deep_learning_project.frameworks.items():
            try:
                # try creating a widget for the framework
                tempWidget = DeepLearningWidget(obj, parent_path = str(self.parent_path), updater = self.update)
                
                # add the widget to the widgets dictionary
                self.deep_learning_widgets[key] = tempWidget

                # add the descriptor of the widget as an item in the drop down
                # (so the user can select the framework they want to use)
                self.network_architecture_drop_down.addItem(obj.descriptor)

                # add the widget for the framework to the stacked widget 
                # (the stacked widget will be displayed when the user selects the framework)
                self.stacked_model_params_layout.addWidget(tempWidget.prediction_widget)
            except Exception as e:
                print(e)
                #self.network_architecture_drop_down.addItem(obj.__name__)
                pass

        # here we use a padding strategy to display the images as a Napari layer in the viewer
        self.images = pad_to_largest(self.deep_learning_project.image_list, force8bit = True) 
        self.viewer.add_image(self.images, name='images')

        self.label_layers_list = []
        self.prediction_layer_list = []

        # add a mark dirty function to indicate that a change is made, we will connect this the events triggerred when layer data is changed 
        def mark_dirty():
            self.dirty = True
            print("Data changed")

        # do the same for labels and predictions
        for c in range(self.deep_learning_project.num_classes):
            temp = pad_to_largest(self.deep_learning_project.annotation_list[c])
            self.viewer.add_labels(temp, name='labels_'+str(c))
            self.label_layers_list.append(self.viewer.layers['labels_'+str(c)])
            self.label_layers_list[c].events.paint.connect(mark_dirty)

            temp = pad_to_largest(self.deep_learning_project.prediction_list[c])
            self.viewer.add_labels(temp, name='predictions_'+str(c))
            self.prediction_layer_list.append(self.viewer.layers['predictions_'+str(c)])

        self.boxes_layer = self.viewer.add_shapes(
            ndim=3,
            name="Label box",
            face_color="transparent",
            edge_color="blue",
            edge_width=5,
        )
        self.boxes_layer.events.data.connect(mark_dirty)

        try:
            self.ml_labels = self.deep_learning_project.ml_labels
            self.ml_features = self.deep_learning_project.ml_features
            self.viewer.add_labels(self.ml_labels, name='ml_labels')
            
            temp = self.deep_learning_project.frameworks["Random Forest Model"]
            temp.create_features(self.images, self.ml_labels, self.ml_features)
        except Exception as e:
            print(f'Error creating ml_labels: {e}')
            print(f'Random Forest ML may not work properly')

        def handle_new_object_box(event):
            self.dirty = True
            # if new box added
            if event.action == 'added':

                # if we have a SAM helper model then use it to segment the box

                    box_ = []
                
                    box = self.object_boxes_layer.data[-1]
                    z = int(box[0,0])
                    
                    # get x and y start and end because this is the format the SAM model expects
                    ystart = int(np.min(box[:,1]))
                    yend = int(np.max(box[:,1]))
                    xstart = int(np.min(box[:,2]))
                    xend = int(np.max(box[:,2]))
                    box_.append([xstart, ystart, xend, yend])
                    box_ = np.array(box_)

                    if self.helper_sam_model is not None:
                        # call the function that segments the bounding box

                        image = self.images[z, :, :]

                        # if image is 2D needs to be pseudo 3D for SAM 
                        if image.ndim == 2:
                            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

                        temp = segment_from_bbox(image, box_, self.helper_sam_model, 'cuda')
                        
                        # get mask and then set the mask pixels that are above 0 to the max value of the labels layer (ie add a new label)
                        temp = temp[0]['segmentation']
                        mask = temp > 0
                        max_ = self.viewer.layers['labels_0'].data[z, :, :].max()
                        self.viewer.layers['labels_0'].data[z, :, :][mask] = max_ + 1
                        self.viewer.layers['labels_0'].refresh()
                    else:
                        roi = np.s_[ystart:yend, xstart:xend]
                        roi2 = np.s_[z, ystart:yend, xstart:xend]
                        pred = self.deep_learning_project.predict_roi(z, self.network_architecture_drop_down.currentText(), self.update, roi)
                                
                        self.label_layers_list[0].data[roi2] = pred
                        self.label_layers_list[0].refresh()              
                    
                #else:
                #    bbox_ = naparixyzbb_to_xyxy(self.object_boxes_layer.data[-1])
                #    print("No SAM model found. Please install segment-everything to use this feature.")

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
                    prediction =  self.prediction_layer_list[c].data[z, ystart:yend, xstart:xend]

                    if np.sum(prediction) > 0:

                        reply = QMessageBox.question(self, 'Overwrite', 'Overwrite existing labels with predictions?', QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

                        if reply == QMessageBox.Yes:
                            # copy from prediction to label
                                self.label_layers_list[c].data[z, ystart:yend, xstart:xend] = self.prediction_layer_list[c].data[z, ystart:yend, xstart:xend]
                                self.label_layers_list[c].refresh()

                                print(self.boxes_layer.data)

        
        print("Adding object boxes layer")

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

        print("Adding predicted object boxes layer")
        
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

        if str(self.mode) != str(EasyAugmentMode.ALL):
            self.viewer.layers.remove('ml_labels')
            self.viewer.layers.remove('Object box')
            self.viewer.layers.remove('Predicted Object box')  

        
        #TODO: integrate to new GUI approach
        '''
        for c in range(self.deep_learning_project.num_classes):
            self.yolo_class_drop_down.addItem("Class "+str(c))
        

        self.object_boxes_layer.feature_defaults['class'] = self.yolo_class_drop_down.currentIndex()
        self.predicted_object_boxes_layer.feature_defaults['class'] = self.yolo_class_drop_down.currentIndex()
        '''        
        self.object_boxes_layer.feature_defaults['class'] = 'c1' 
        self.predicted_object_boxes_layer.feature_defaults['class'] = 'c1' 

        print("Adding label boxes")
        if self.deep_learning_project.boxes is not None:
            self.boxes_layer.add(self.deep_learning_project.boxes)

        print("Adding object boxes")
        if self.deep_learning_project.object_boxes is not None:
            self.object_boxes_layer.add(self.deep_learning_project.object_boxes)

        print("Adding predicted object boxes")
        if self.deep_learning_project.predicted_object_boxes is not None:
            self.predicted_object_boxes_layer.add(self.deep_learning_project.predicted_object_boxes)

        print("Setting object box classes")
        if self.deep_learning_project.classes is not None:
            self.object_boxes_layer.features = self.deep_learning_project.classes

        print("Setting predicted object box classes")
        if self.deep_learning_project.predicted_classes is not None:
            self.predicted_object_boxes_layer.features = self.deep_learning_project.predicted_features
        
        self.boxes_layer.events.data.connect(handle_new_roi)
        self.object_boxes_layer.events.data.connect(handle_new_object_box)

        self.dirty = False
   
    def set_pretrained_model(self, model_path, model_type):
        widget = self.deep_learning_widgets[model_type]
        widget.load_model_from_path(model_path) 

    def save_results(self):
        self.update_annotation_list()

        object_boxes=self.object_boxes_layer.data
        
        self.deep_learning_project.save_project(self.viewer.layers['Label box'].data, self.save_histogram_check_box.isChecked()) 

        if len(object_boxes)>0:        
            object_classes = self.object_boxes_layer.features['class'].to_numpy()
        
            self.deep_learning_project.save_object_boxes(object_boxes, object_classes)

        # Should we show this box?
        #QMessageBox.information(self, "Save Results", "Results saved successfully.")

        self.dirty = False

    def check_labels(self):

        # check if any boxes have been drawn if not we have no labels
        if len(self.boxes_layer.data) == 0:
            QMessageBox.information(self, "Error", "No label boxes drawn yet.")
            return False

        return self.deep_learning_project.check_labels(self.boxes_layer.data)
    
    def ask_about_labels(self):
        reply = QMessageBox.question(
            self,
            "Are labels updated?",
            "Do you want to augment using current label boxes and labels?  If not press No and then continue labelling",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes)

        return reply
    
    
    def augment_current(self):
        
        # check if we have labels drawn
        labels_ok = self.check_labels()

        if not labels_ok:
            return

        # if we have labels drawn nag the user to check if they are good to go        
        reply = self.ask_about_labels()
        
        if reply == QMessageBox.No:
            return
        
        self.update_annotation_list()
        
        self.textBrowser_log.append("Augmenting current image...")

        n = self.viewer.dims.current_step[0]

        # get current boxes at n
        boxes = self.viewer.layers['Label box'].data

        boxes = np.array(boxes)
        index_boxes = np.all(boxes[:,:,0]==n, axis=1)
        filtered_boxes = boxes[index_boxes]

        # check if any boxes have been drawn for the current image, if not we have no labels
        if filtered_boxes.shape[0] == 0:
            QMessageBox.information(self, "Error", f"No label boxes drawn for image {n} yet.")
            return
        
        self.save_results()
        self.perform_augmentation(filtered_boxes)

    def augment_all(self):
        
        # check if we have labels drawn
        labels_ok = self.check_labels()

        if not labels_ok:
            return

        # if we have labels drawn nag the user to check if they are good to go        
        reply = self.ask_about_labels()
        
        if reply == QMessageBox.No:
            return
        
        self.update_annotation_list()
        
        boxes = self.viewer.layers['Label box'].data
        self.save_results()
        self.perform_augmentation(boxes)

    def delete_augmentations(self):
        self.deep_learning_project.delete_augmentations()

    def augment_settings(self):
        dialog = QDialog()

        size_factor = self.deep_learning_project.augmentation_parameters['size_factor']
        alpha = self.deep_learning_project.augmentation_parameters['alpha']
        sigma = self.deep_learning_project.augmentation_parameters['sigma']
        alpha_affine = self.deep_learning_project.augmentation_parameters['alpha_affine']
        hue = self.deep_learning_project.augmentation_parameters['hue']
        brightness = self.deep_learning_project.augmentation_parameters['brightness']
        saturation = self.deep_learning_project.augmentation_parameters['saturation']
        normalization_type = self.deep_learning_project.augmentation_parameters['normalization_type']

        # Rescale group
        rescale_label = QLabel("Rescale")
        size_factor_spinner = LabeledSpinner("Size Factor", 0.1, 10, size_factor, None, is_float=True, step=0.1)
        size_factor_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("size_factor", value))

        # Elastic group
        elastic_label = QLabel("Elastic")
        alpha_spinner = LabeledSpinner("Alpha", 0.01, 100, alpha, None, is_float=True, step=1)
        sigma_spinner = LabeledSpinner("Sigma", 0.1, 10, sigma, None, is_float=True, step=0.1)
        alpha_affine_spinner = LabeledSpinner("Alpha Affine", 0.1, 10, alpha_affine, None, is_float=True, step=0.1)

        alpha_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("alpha", value))
        sigma_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("sigma", value))
        alpha_affine_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("alpha_affine", value))

        # color group (Hue, Brightness, Saturation)
        color_label = QLabel("Color")
        hue_spinner = LabeledSpinner("Hue", 0.0, 0.5, hue, None, is_float=True, step=0.01)
        brightness_spinner = LabeledSpinner("Brightness", 0.0, 0.5, brightness, None, is_float=True, step=0.01)
        saturation_spinner = LabeledSpinner("Saturation", 0.0, 0.5, saturation, None, is_float=True, step=0.01)
        hue_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("hue", value))
        brightness_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("brightness", value))
        saturation_spinner.spinner.valueChanged.connect(lambda value: self.deep_learning_project.set_augmentation_parameter("saturation", value))

        # misc group (image vs label normalization and maybe more)
        misc_label = QLabel("Misc")

        def normalization_type_changed(value):
            if value == 0:
                self.deep_learning_project.set_augmentation_parameter("normalization_type", "label")
            elif value == 1:
                self.deep_learning_project.set_augmentation_parameter("normalization_type", "image")
       
        normalization_type_combo = LabeledCombo("Normalization Type", ["Just Label", "Entire Image"], normalization_type_changed)
        normalization_type_combo.combo.setCurrentIndex(0 if normalization_type == "label" else 1)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(dialog.accept)

        # Layout
        dialog_layout = QVBoxLayout()
        dialog_layout.addWidget(rescale_label)
        dialog_layout.addWidget(size_factor_spinner)
        dialog_layout.addWidget(elastic_label)
        dialog_layout.addWidget(alpha_spinner)
        dialog_layout.addWidget(sigma_spinner)
        dialog_layout.addWidget(alpha_affine_spinner)
        dialog_layout.addWidget(color_label)
        dialog_layout.addWidget(hue_spinner)
        dialog_layout.addWidget(brightness_spinner)
        dialog_layout.addWidget(saturation_spinner)
        dialog_layout.addWidget(misc_label)
        dialog_layout.addWidget(normalization_type_combo)
        dialog_layout.addWidget(ok_button)

        dialog.setLayout(dialog_layout)
        dialog.exec_()
        
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

        model_name = self.network_architecture_drop_down.currentText()


        if self.deep_learning_project.frameworks[model_name].boxes ==True:

            # Check if no boxes are drawn
            if len(objects) == 0:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setText("No boxes drawn yet.")
                msg.setInformativeText("Please draw at least one box before performing augmentation.")
                msg.setWindowTitle("No Boxes Found")
                msg.exec_()
                return

            classes = self.object_boxes_layer.features['class'].to_numpy()
            self.deep_learning_project.perform_yolo_augmentation(boxes, objects, classes, num_patches_per_roi, patch_size, self.update,
                                                                 perform_horizontal_flip, perform_vertical_flip, perform_random_rotate, perform_random_resize, 
                                                                 perform_random_brightness_contrast, perform_random_gamma, perform_random_adjust_color)
        else:

            self.deep_learning_project.perform_augmentation(boxes, num_patches_per_roi, patch_size, self.update,
                                                                 perform_horizontal_flip, perform_vertical_flip, perform_random_rotate, perform_random_resize, 
                                                                 perform_random_brightness_contrast, perform_random_gamma, perform_random_adjust_color, perform_elastic_deformation)

    def perform_training(self):

        patches_ok = self.deep_learning_project.check_patches()
        if not patches_ok:
            QMessageBox.warning(self, "Error", "Patches not valid (do not exist or are inconsistent). Please augment the images and/or double check augmentations before training.")
            return
        
        reply = QMessageBox.question(
            self,
            "Augmentations updated?",
            "Do you want to train on current augmented patches?  If not press No and then continue labelling and augmenting",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.No:
            return

        widget = self.deep_learning_widgets[self.network_architecture_drop_down.currentText()]
        dialog = widget.train_dialog
        dialog.exec_()

        # TODO: work out how to handle augmentation before training
        # perform another round of augmentation before training
        # The user also has the option to perform augmentation before training as to customize the augmented data
        # (for example they can augment a single image multiple times potentially with different augmentations, weighting that image more)
        # So ideally the user would augment the data before training
        # However this will confuse some users, as if they don't augment, not all images will be used in training
        # Thus we do a final round of augmentation before training?
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
            self.worker = PyQt5WorkerThread(self.deep_learning_project.perform_training, self.network_architecture_drop_down.currentText(), None)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.disable_gui)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(self.update)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.enable_gui)
            self.thread.finished.connect(self.training_finished)

            self.thread.start()
             
        else:
            self.deep_learning_project.perform_training(self.network_architecture_drop_down.currentText(), self.update)
            widget.sync_with_framework()

    def training_finished(self):
        self.enable_gui()
        widget = self.deep_learning_widgets[self.network_architecture_drop_down.currentText()]
        widget.sync_with_framework()

    def predict_current_image(self):
        self.textBrowser_log.append("Predicting current image...")
        n = self.viewer.dims.current_step[0]
        
        model_name = self.network_architecture_drop_down.currentText()
        show_boxes = self.deep_learning_project.frameworks[model_name].boxes
       
        if show_boxes == True:
            predictions, boxes = self.deep_learning_project.predict(n, model_name, self.update)
            
            #self.object_boxes_layer.add(boxes)
            #self.object_boxes_layer.refresh()

            self.predicted_object_boxes_layer.add(boxes)
            self.predicted_object_boxes_layer.refresh()

            self.prediction_layer_list[0].data[n, :predictions.shape[0], :predictions.shape[1]]=predictions
            self.prediction_layer_list[0].refresh() 
        else:
            
            pred = self.deep_learning_project.predict(n, self.network_architecture_drop_down.currentText(), self.update)
                    
            self.prediction_layer_list[0].data[n, :pred.shape[0], :pred.shape[1]]=pred
            self.prediction_layer_list[0].refresh()              
        
            pass
    
    def predict_all_images(self):
        
        self.textBrowser_log.append("Predicting all images...")

        model_name = self.network_architecture_drop_down.currentText()
        show_boxes = self.deep_learning_project.frameworks[model_name].boxes
        
        thread = True

        if thread:
            
            self.thread = QThread()
            self.worker = PyQt5WorkerThread(self.deep_learning_project.predict_all, model_name, self.update_thread)
            self.worker.moveToThread(self.thread)
            self.thread.started.connect(self.disable_gui)
            self.thread.started.connect(self.worker.run)
            self.worker.progress.connect(self.update)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.finished.connect(self.enable_gui)
            self.thread.finished.connect(lambda: self.predict_all_images_finished(show_boxes))
            self.thread.start()
       
        else:
            self.deep_learning_project.predict_all(model_name, self.update)
            self.predict_all_images_finished(show_boxes)

    def predict_all_images_finished(self, show_boxes):
        
        predictions = pad_to_largest(self.deep_learning_project.prediction_list[0])
        self.prediction_layer_list[0].data = predictions
        
        if show_boxes == True:
            self.predicted_object_boxes_layer.add(self.deep_learning_project.predicted_object_boxes)

    def disable_gui(self):
        self.setEnabled(False) 

    def enable_gui(self):
        self.setEnabled(True)

    def update_annotation_list(self):   
        annotation_list = []

        # loop through all label layers (will be one for each class) 
        # and unpad the data to the original size of the image
        # this is because the data is padded to the largest image size when displayed in Napari 
        for label_layer in self.label_layers_list:
            temp = unpad_to_original(label_layer.data, self.deep_learning_project.image_list)
            annotation_list.append(temp)

        # update annotation list on the deep_learning_project side
        self.deep_learning_project.annotation_list = annotation_list
    
    def hideEvent(self, event):

        if self.dirty:
            reply = QMessageBox.question(
                self,
                "Easy Augment Save Changes?",
                "Easy-Augment-Batch-DL has unsaved changes. Do you want to save before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )

            if reply == QMessageBox.Cancel:
                event.ignore()  # Don't close the widget
            elif reply == QMessageBox.Yes:
                self.save_results()
                event.accept()
            else:
                event.accept()
        else:
            event.accept()

       