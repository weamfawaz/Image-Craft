import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
import tensorflow as tf 
import os 
from PIL import Image
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.python.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization 
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import GlobalAveragePooling2D 
 
#Defining size 
batch_size = 32 
img_height = 150 
img_width = 150 
 # Get the current working directory
cwd = os.getcwd()
train_folder= os.path.join(cwd, 'seg_train')
train_ds = tf.keras.utils.image_dataset_from_directory( 
  train_folder, 
  seed=123, 
  image_size=(img_height, img_width), 
  batch_size=batch_size)

valid_folder=os.path.join(cwd, 'seg_test')
val_ds = tf.keras.utils.image_dataset_from_directory(
  valid_folder,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

#Labels
label_to_class_name = dict(zip(range(len(train_ds.class_names)), train_ds.class_names))

data_iterator = train_ds.as_numpy_iterator()
batch = data_iterator.next()
fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        index = i * 4 + j
        ax[i, j].imshow(batch[0][index].astype(int))
        ax[i, j].set_title(label_to_class_name[batch[1][index]])
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

#Size 
for image_batch, labels_batch in train_ds: 
  print(image_batch.shape) 
  print(labels_batch.shape) 
  break


# Resize images to (224, 224)
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))
val_ds = val_ds.map(lambda x, y: (tf.image.resize(x, (224, 224)), y))

# Normalize pixel values to [0, 1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))
# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE 
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE) 
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
# Load MobileNetV2 model with modified input shape
pretrained_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    input_shape=(224, 224, 3),  # Modified input shape
    pooling='max',
    classes=6, 
    weights='imagenet'
)
mobile_net = Sequential() 
mobile_net.add(pretrained_model) 
mobile_net.add(Flatten()) 
mobile_net.add(Dense(512, activation='relu')) 
mobile_net.add(BatchNormalization())  # Batch Normalization layer 
mobile_net.add(Dropout(0.5)) 
mobile_net.add(Dense(6, activation='softmax')) 
pretrained_model.trainable = False
mobile_net.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
) 
mobile_net.summary()

# Train the model
epochs = 1 
history = mobile_net.fit( 
    train_ds, 
    validation_data=val_ds, 
    epochs=epochs 
)
loss = history.history['loss'] 
val_loss = history.history['val_loss'] 
epochs = range(1, len(loss)+1) 
plt.plot(epochs, loss, 'b', label='Training loss') 
plt.plot(epochs, val_loss, 'r', label='Validation loss') 
plt.title('Training and Validation Loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
plt.show() 

acc = history.history['accuracy'] 
val_acc = history.history['val_accuracy'] 
plt.plot(epochs, acc, 'b', label='Training acc') 
plt.plot(epochs, val_acc, 'r', label='Validation acc') 
plt.title('Training and Validation Accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()

# Save the model in the native Keras format with .keras extension
mobile_net.save('mobilenet_natural_images.keras')


# Load the model
loaded_model = tf.keras.models.load_model('mobilenet_natural_images.keras')

# Use the loaded model for predictions
img = cv2.imread('/seg_pred/10012.jpg') 
plt.imshow(img) 
plt.show() 
# Resize and preprocess the image
resize = cv2.resize(img, (224, 224))  # Resize the image
resize = resize / 255.0  # Normalize pixel values to [0, 1]
 
# Make predictions
yhat = loaded_model.predict(np.expand_dims(resize, axis=0))
max_index = np.argmax(yhat)
predicted_class = label_to_class_name[max_index]
print("Predicted class:", predicted_class)

