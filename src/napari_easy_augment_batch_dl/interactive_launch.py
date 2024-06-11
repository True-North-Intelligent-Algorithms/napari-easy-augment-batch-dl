import napari

from napari_easy_augment_batch_dl import easy_augment_batch_dl

viewer = napari.Viewer()

viewer.window.add_dock_widget(
    easy_augment_batch_dl.NapariEasyAugmentBatchDL(viewer)
)
napari.run()
