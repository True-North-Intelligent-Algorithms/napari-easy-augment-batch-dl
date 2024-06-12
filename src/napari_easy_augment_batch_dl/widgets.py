# qypt imports
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
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
