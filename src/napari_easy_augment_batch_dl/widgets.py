# qypt imports
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
    QComboBox,
    QLineEdit
)

class LabeledSpinner(QWidget):
    def __init__(self, label_text, min_value, max_value, default_value, change_value_method, is_double=False, step = -1):
        super().__init__()

        self.label = QLabel(label_text)
        self.spinner = QDoubleSpinBox() if is_double else QSpinBox()
        self.spinner.setRange(min_value, max_value)
        self.spinner.setValue(default_value)

        if change_value_method is not None:
            self.spinner.valueChanged.connect(change_value_method)

        if step > 0:
            self.spinner.setSingleStep(step)
            if is_double:
                self.spinner.setDecimals(3)

        self.layout = QHBoxLayout()
        self.layout.setContentsMargins(10, 2, 10, 2)  # adjust the margins around the layout
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spinner)

        self.setLayout(self.layout)

    def setValue(self, value):
        self.spinner.setValue(value)

class LabeledCombo(QWidget):
    def __init__(self, label_text, items, change_value_method):
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