from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QWidget, QFileDialog, QCheckBox
from napari_easy_augment_batch_dl.widgets import LabeledSpinner, LabeledCombo, LabeledEdit
from napari_easy_augment_batch_dl.frameworks.base_framework import LoadMode
from PyQt5.QtCore import Qt  # This brings in the Qt constants
import os

class DeepLearningWidget(QDialog):
    """
    A widget for setting deep learning parameters in a deep learning framework.
    """
    
    def __init__(self, framework, parent=None, parent_path=None, updater=None):
        super(DeepLearningWidget, self).__init__(parent)

        # parent path of the project where images and deep learning artifacts are stored
        self.framework = framework 
        self.parent_path = parent_path
        self.updater = updater

        if self.updater is None:
            self.updater = lambda x: print(x)

        # collect all the fields that are marked for harvesting to the gui
        self.harvested_params = {name: field.metadata for name, field in framework.__dataclass_fields__.items() if field.metadata.get('harvest')}
        
        # get the existing model names and optimizers
        self.model_names = framework.get_model_names()
        self.optimizers = framework.get_optimizers()
        
        # create layout for prediction parameters
        self.prediction_layout = QVBoxLayout()
        self.prediction_widget = QWidget()
        
        # create layout for training parameters
        self.train_layout = QVBoxLayout()
        self.train_widget = QWidget()

        # Create input fields based on the harvested parameters
        self.fields = {}
        # loop through the harvested params and create a widget for each
        for param_name, meta in self.harvested_params.items():

            train = meta.get('training', False)
            
            if meta['type'] == 'int':
                min = meta.get('min', 0)
                max = meta.get('max', 1)
                default = meta.get('default', 0)
                step = meta.get('step', 1)
                field = LabeledSpinner(param_name, min, max, default, None, is_float=False, step=step)
                field.spinner.valueChanged.connect(lambda value, name=param_name: setattr(self.framework, name, value))
            elif meta['type'] == 'float':
                min = meta.get('min', 0)
                max = meta.get('max', 1)
                default = meta.get('default', 0.5)
                step = meta.get('step', 0.1)
                field = LabeledSpinner(param_name, min, max, default, None, is_float=True, step=step)
                field.spinner.valueChanged.connect(lambda value, name=param_name: setattr(self.framework, name, value))
            elif meta['type'] == 'string':
                field = QLineEdit()
                field.setPlaceholderText("Enter a string")
            elif meta['type'] == 'bool':
                field = QCheckBox(param_name)
                field.setChecked(meta.get('default', False))
                field.stateChanged.connect(lambda state, name=param_name: setattr(self.framework, name, state == Qt.Checked))
            else:
                # see if model has param name
                if hasattr(self.framework, param_name):
                    default = getattr(self.framework, param_name)
                    place_holder = None
                else:
                    default = None
                    place_holder = "Enter a string"

                field = LabeledEdit(param_name, default, place_holder, None)
                field.edit.textChanged.connect(lambda text, name=param_name: setattr(self.framework, name, text))

            self.fields[param_name] = field
            
            if train:
                self.train_layout.addWidget(field)            
            else:
                self.prediction_layout.addWidget(field)

        # the model name is a special case.  We need a combo box and special handler to select the model
        if len(self.model_names) > 0:
            # add combo with model names
            self.pretrained_combo = LabeledCombo('Model', self.model_names, self.on_model_name_change)
            self.prediction_layout.addWidget(self.pretrained_combo)

        # same with the optimizers
        if len(self.optimizers) > 0:
            self.optimizer_combo = LabeledCombo('Optimizer', self.optimizers, self.on_optimizer_change)
            self.train_layout.addWidget(self.optimizer_combo)

        # also need to add a load button if the model is loadable    
        if self.framework.load_mode == LoadMode.Directory or self.framework.load_mode == LoadMode.File:
            # add load button
            load_button = QPushButton("Load")
            load_button.clicked.connect(self.show_load_model_dialog)
            self.prediction_layout.addWidget(load_button)

        # add the prediction and training layouts to the widgets
        self.prediction_widget.setLayout(self.prediction_layout)
        self.train_widget.setLayout(self.train_layout)

        # create a dialog for the training parameters 
        self.train_dialog = QDialog()
        layout = QVBoxLayout(self.train_dialog)
        layout.addWidget(self.train_widget)
        self.train_dialog.setLayout(layout)
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.train_dialog.accept)
        layout.addWidget(ok_button)
        self.train_dialog.setAttribute(Qt.WA_DeleteOnClose, False)

    def on_model_name_change(self, index):
        """
        Callback for when the model name is changed in the combo box.
        """
        model_name = self.model_names[index]
        self.framework.set_pretrained_model(model_name)

    def on_optimizer_change(self, index):
        """
        Callback for when the optimizer is changed in the combo box.
        """
        optimizer = self.optimizers[index]
        self.framework.set_optimizer(optimizer)
    
    def show_load_model_dialog(self):
        """
        Show a dialog for selecting a model file or directory.
        """

        # if the load mode is file, show a file dialog, otherwise show a directory dialog
        if self.framework.load_mode == LoadMode.File:
            options = QFileDialog.Options()
            file_, _ = QFileDialog.getOpenFileName(self, "Select Model File", self.parent_path, "All Files (*)", options=options)
        else:
            options = QFileDialog.Options()
            file_ = QFileDialog.getExistingDirectory(self, "Select Model Directory", options=options, directory = self.parent_path)

        if len(file_) > 0:        
            self.updater(f"Loading model from {os.path.basename(file_)}")
            self.load_model_from_path(file_)
        else:
            self.updater("No file selected")

    def load_model_from_path(self, file_):
        """
        Load a model from the path (can be a file or directory).
        """ 

        # call the method in the framework to load the model
        self.framework.load_model_from_disk(file_)

        # model name is the base name of the file
        model_name = os.path.basename(file_)
        
        # add model name to the combo and edit field
        self.pretrained_combo.combo.addItem(model_name)
        self.fields['model_name'].edit.setText(model_name)
        self.model_names.append(model_name)
        
        # set the combo to the new model name
        self.pretrained_combo.combo.setCurrentIndex(self.pretrained_combo.combo.count()-1)

    def sync_with_framework(self):
        """
        Sync the widget with the framework.
        This is used when the parameters are changed outside of the widget.
        """
        try:
            # loop throush the harvested params and set the values in the gui fields
            # (this code is a bit abstract but remember we added widgets for each field and stored them in self.fields)
            for param_name, meta in self.harvested_params.items():
                if hasattr(self.framework, param_name):
                    value = getattr(self.framework, param_name)
                    self.fields[param_name].setValue(value)
                else:
                    self.fields[param_name].setText("")
        except Exception as e:
            print(e)

        try:
            # get model names in combo
            model_names = [self.pretrained_combo.combo.itemText(i) for i in range(self.pretrained_combo.combo.count())]
            
            # get current model name
            model_name = self.framework.model_name
            
            # if current model name isn't in combo, add it
            if model_name not in model_names:
                self.pretrained_combo.combo.addItem(model_name)
                self.model_names.append(model_name)
            
            self.pretrained_combo.combo.setCurrentText(model_name)
        except Exception as e:
            print(e)
