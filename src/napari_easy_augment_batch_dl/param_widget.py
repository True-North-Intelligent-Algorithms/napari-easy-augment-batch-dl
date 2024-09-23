from qtpy.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QWidget, QFileDialog
from napari_easy_augment_batch_dl.widgets import LabeledSpinner, LabeledCombo
from napari_easy_augment_batch_dl.base_model import LoadMode

class ParamWidget(QDialog):
    def __init__(self, model, parent=None):
        super(ParamWidget, self).__init__(parent)

        self.model = model 
        self.harvested_params = {name: field.metadata for name, field in model.__dataclass_fields__.items() if field.metadata.get('harvest')}
        self.model_names = model.get_model_names()
        self.layout = QVBoxLayout()
        self.widgetGroup = QWidget()
        
        # Create input fields based on the harvested parameters
        self.fields = {}
        for param_name, meta in self.harvested_params.items():
            #label = QLabel(param_name)
            #layout.addWidget(label)
            
            if meta['type'] == 'int':
                min = meta.get('min', 0)
                max = meta.get('max', 1)
                default = meta.get('default', 0)
                step = meta.get('step', 1)
                field = LabeledSpinner(param_name, min, max, default, None, is_double=False, step=step)
                field.spinner.valueChanged.connect(lambda value, name=param_name: setattr(self.model, name, value))
            elif meta['type'] == 'float':
                #field = QLineEdit()
                #field.setPlaceholderText("Enter a float")
                min = meta.get('min', 0)
                max = meta.get('max', 1)
                default = meta.get('default', 0.5)
                step = meta.get('step', 0.1)
                field = LabeledSpinner(param_name, min, max, default, None, is_double=True, step=step)
                field.spinner.valueChanged.connect(lambda value, name=param_name: setattr(self.model, name, value))
            elif meta['type'] == 'string':
                field = QLineEdit()
                field.setPlaceholderText("Enter a string")
            else:
                field = QLineEdit()
                
            self.fields[param_name] = field
            self.layout.addWidget(field)

            # add call back to set the value
            #field.textChanged.connect(lambda text, name=param_name: setattr(self.model, name, text))


        if len(self.model_names) > 0:
            # add combo with model names
            '''
            label = QLabel('Model')
            self.layout.addWidget(label)
            self.network_architecture_drop_down = QComboBox()
            self.network_architecture_drop_down.addItems(self.model_names)
            self.layout.addWidget(self.network_architecture_drop_down)
            self.network_architecture_drop_down.currentIndexChanged.connect(self.on_model_name_change)
            '''

            self.pretrained_combo = LabeledCombo('Model', self.model_names, self.on_model_name_change)
            self.layout.addWidget(self.pretrained_combo)

        if self.model.load_mode == LoadMode.Directory:
            # add load button
            load_button = QPushButton("Load")
            load_button.clicked.connect(self.load_model)
            self.layout.addWidget(load_button)
        elif self.model.load_mode == LoadMode.File:
            # add load button
            load_button = QPushButton("Load")
            load_button.clicked.connect(self.load_model)
            self.layout.addWidget(load_button)

       
        # Add OK button to close the dialog
        #ok_button = QPushButton("OK")
        #ok_button.clicked.connect(self.accept)
        #layout.addWidget(ok_button)
        
        self.widgetGroup.setLayout(self.layout)
    
    def on_model_name_change(self, index):
        model_name = self.model_names[index]
        self.model.set_pretrained_model(model_name)
    
    def get_values(self):
        # Return the values entered in the dialog
        return {name: field.text() for name, field in self.fields.items()}
    
    def load_directory(self):
        pass;

    def load_model(self):

        if self.model.load_mode == LoadMode.File:
            options = QFileDialog.Options()
            file_, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "All Files (*)", options=options)
        else:
            options = QFileDialog.Options()
            file_ = QFileDialog.getExistingDirectory(self, "Select Model Directory", options=options)
        
        self.model.load_model_from_disk(file_)

        model_name = file_.split('/')[-1]

        self.pretrained_combo.combo.addItem(model_name)
        self.model_names.append(model_name)
        self.pretrained_combo.combo.setCurrentIndex(self.pretrained_combo.combo.count()-1)

