# Train and Predict  

Train a model using your labeled data and make predictions on new images.  

## 🏋️ Training & Prediction Panel  
![Training Panel](images/train_panel.png)  

### Steps:  
1️⃣ Choose a model from dropdown and configure training parameters.  
2️⃣ Train using labeled images.  
3️⃣ Use the trained model to predict labels on new images.  

## Training popup 

After hitting train a popup should appear which allows you to further adjust training parameters.  

![Training popup](images/train_cellpose.png)

After training (or after loading or setting a model) choose ```Predict current image``` or ```Predict all images``` to predict.   The ```prediction``` layer should be populated with the predictions as shown in the below screen shot.  

![Prediction](images/predict.png)

After predicting you need to save the project again and the predictions will be written to disk

![Save results...](images/save_results.png)

The predictions will be written in your project folder under ```predictions\class_0```.  

![Saved predictions](images/saved_predictions.png)

---

🔄 **Next:** [Run & Export](run_and_export.md)  
