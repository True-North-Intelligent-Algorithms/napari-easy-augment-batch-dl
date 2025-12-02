from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QFileDialog, QMessageBox
from napari_easy_augment_batch_dl import easy_augment_batch_dl
import sys
import napari
import torch
import cellpose

# print versions
print("Napari version:", napari.__version__)
print("PyTorch version:", torch.__version__)
print("Cellpose version:", cellpose.version)

# Check if CUDA (GPU) is available
print("CUDA available:", torch.cuda.is_available())

# If available, print the name of the GPU
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())
else:
    print("Running on CPU")

viewer = napari.Viewer()

# create instance of NapariEasyAugmentBatchDL
batch_dl = easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer)

# Wrap in a scroll area to ensure full visibility
class WideScroll(QtWidgets.QScrollArea):
    def sizeHint(self):
        return QtCore.QSize(500, 300)  # width=900, height arbitrary

scroll = WideScroll()
scroll.setWidgetResizable(True)
scroll.setWidget(batch_dl)

viewer.window.add_dock_widget(
    scroll
)

# Popup to select parent path
parent_path = QFileDialog.getExistingDirectory(
    None,
    "Please select parent directory (project folder with images)",
    "",
    QFileDialog.ShowDirsOnly
)

# load images if a path was selected
if parent_path:
    print(f"Selected parent path: {parent_path}")
    batch_dl.load_image_directory(parent_path)
else:
    QMessageBox.warning(None, "No Directory Selected", "No parent directory was selected. Please use the plugin to load images.")
    print("No parent path selected - skipping load")

print("Setting framework to Micro-sam Instance Framework")
framework_type = "Micro-sam Instance Framework"
batch_dl.network_architecture_drop_down.setCurrentText(framework_type)

napari.run()
