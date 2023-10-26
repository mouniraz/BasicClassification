# BasicClassification
## Load and explore Data
1. load data from tensorflow datasets
   ```python
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
   ```
2. explore data and display shape and dim of those four arrays
3. plot first images in train_models and see the range of pixels
```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```
4. scale images to be between 0 and 1 (deviding by 255)
   ```python
   train_images = train_images / 255.0
   test_images = test_images / 255.0
   ```
5. plot the 20 first images 
## Compile and Fit data
1. compile and train your model
## Evaluate unseen data
1. evaluate with test data
2. there is overfitting in this trained model, explain 
## make predictions
1. make predictions of test_images and if prediction is correct 
   ```python
   probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
   predictions = probability_model.predict(test_images)
   predictions[0]
   np.argmax(predictions[0])
   ```
2. display image and prediction
   ```python
   i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
   ``` 
