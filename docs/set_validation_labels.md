# Set Validation Labels

## Training vs Validation Labels

By default, all labels you draw are **training labels**. These are used to train the model. However, it's important to also set aside some **validation labels** to monitor model performance during training.

**Training Labels:**
- Used to teach the model patterns and features
- The model learns from these labels
- Updated during each training iteration

**Validation Labels:**
- Used to evaluate model performance during training
- NOT used for learning - kept separate to test generalization
- Helps detect overfitting and monitor training progress

**Important:** Not all frameworks use validation labels. Check your framework documentation or ask on [image.sc](https://forum.image.sc/) to see if validation is supported.

## How to Set Validation Labels

![Sparse Train Validation Label](images/sparse_train_validation_label.png)

1️⃣ Select the **Label box** layer in the layer list  
2️⃣ Click the **selection tool** (hollow arrow) in Napari's left toolbar  
3️⃣ Select the label box ROIs that you want to designate as validation  
4️⃣ Press the **`v`** key to toggle between training and validation  

The text on the label box will change to indicate its type (training or validation).

## Outputs

In the json file associated with the label you should now see a "name -> value' pair with name "split" and value either "train" or "validation".  If you use the labels downstream of ```Napari-easy-augment-batch-dl``` then use this name value pair when partitioning into train/validation sets. 

![Label JSON](images/label_json.png)

## Training and validation patches

If you set some labels as ```validation```, the patches generated from those labels will be partitioned into a separate directory (with ```_validation``` partitioned on the directory name).  See screenshot below...

![Validation Labels](images/train_validation_patches.png)

## Notes

Not all frameworks support validation labels yet.  

## Best Practices

- **Reserve 10-20%** of your labeled data for validation
- Ensure validation data represents the **variety** in your full dataset
- Validation labels should be **independent** from training labels
- Toggle validation labels **before** running augmentation and training

---

🔄 **Next:** [Configure Augmentations](augment.md)
