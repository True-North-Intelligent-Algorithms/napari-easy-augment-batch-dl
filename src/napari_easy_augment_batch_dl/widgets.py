# qypt imports
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
    QComboBox,
    QLineEdit,
    QCheckBox,
)

class LabeledSpinner(QWidget):
    def __init__(self, label_text, min_value, max_value, default_value, change_value_method, is_float=False, step = -1, num_decimals = 5, show_auto_checkbox=False):
        super().__init__()

        self.label = QLabel(label_text)
        self.spinner = QDoubleSpinBox() if is_float else QSpinBox()
        self.spinner.setRange(min_value, max_value)

        if change_value_method is not None:
            self.spinner.valueChanged.connect(change_value_method)

        if step > 0:
            self.spinner.setSingleStep(step)
        if is_float:
            self.spinner.setDecimals(num_decimals)
        
        self.spinner.setValue(default_value)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout

        # Create a sub-layout for the label and optional checkbox
        self.label_layout = QHBoxLayout()

        self.label_layout.addWidget(self.label)

        # (Optional) Create a checkbox to toggle auto-calculation
        # Sometimes parameters are automatically calculated if not passed in to a function or if None
        # in these cases we want to avoid passing a value to the function and disable the spinner
        if show_auto_checkbox:
            self.auto_checkbox = QCheckBox("Auto")
            self.auto_checkbox.setChecked(True)
            self.auto_checkbox.stateChanged.connect(self.toggle_spinner)
            self.auto_checkbox.setToolTip("Enable auto-calculation")
            self.spinner.setEnabled(False)
            self.label_layout.addWidget(self.auto_checkbox)
        
        self.layout.addLayout(self.label_layout)
        self.layout.addWidget(self.spinner)

        self.setLayout(self.layout)

    def setValue(self, value):
        self.spinner.setValue(value)

    def toggle_spinner(self, state):
        """Enable spinner when auto checkbox is unchecked, disable otherwise."""
        self.spinner.setEnabled(state == 0)

class LabeledCombo(QWidget):
    def __init__(self, label_text, items, change_value_method=None):
        super().__init__()

        self.label = QLabel(label_text)
        self.combo = QComboBox()
        self.combo.addItems(items)

        if change_value_method is not None:
            self.combo.currentIndexChanged.connect(change_value_method)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.combo)

        self.setLayout(self.layout)

    def setValue(self, value):
        index = self.combo.findText(value)
        if index >= 0:
            self.combo.setCurrentIndex(index)

class LabeledEdit(QWidget):
    def __init__(self, label_text, default_text, place_holder_text, change_value_method):
        super().__init__()

        self.label = QLabel(label_text)
        self.edit = QLineEdit()

        if default_text is not None:
            self.edit.setText(default_text)
        else:
            self.edit.setPlaceholderText(place_holder_text)
        
        if change_value_method is not None:
            self.edit.textChanged.connect(change_value_method)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.edit)

        self.setLayout(self.layout)

    def setValue(self, value):
        self.edit.setText(value)