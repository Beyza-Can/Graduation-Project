import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

import os

# Folders
train_folder = "image-Data/train"
valid_folder = "image-Data/valid"
test_folder = "image-Data/test"

input_shape = (299, 299, 3)  # Xception input shape
num_classes = 4

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)
test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(299, 299),  # Xception input size
    batch_size=64,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(299, 299),  # Xception input size
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

validation_generator = val_datagen.flow_from_directory(
    valid_folder,
    target_size=(299, 299),  # Xception input size
    batch_size=64,
    class_mode='categorical'
)

# Load Xception model pretrained on ImageNet without including top layers
base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom top layers for your task
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint = ModelCheckpoint('best_model_xception.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='min')

epochs = 50

# Model training
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[checkpoint, early_stopping, reduce_lr],
    verbose=1
)

# Save the final model
model.save('final_model_xception.keras')

# Model evaluation
test_loss, test_accuracy = model.evaluate(test_generator)
print('Test accuracy:', test_accuracy)

predicted_probabilities = model.predict(test_generator)
predicted_labels = np.argmax(predicted_probabilities, axis=1)
true_labels = test_generator.classes

cm = confusion_matrix(true_labels, predicted_labels)
print("\nConfusion Matrix:\n", cm)

plt.figure(figsize=(10, 4))
sns.heatmap(cm, annot=True, fmt='g', cmap='Reds')
plt.xlabel('\nPredicted Label\n')
plt.ylabel('\nTrue Label\n')
plt.title('Confusion Matrix\n')
plt.show()

report = classification_report(true_labels, predicted_labels)
print(report)

# Plot training & validation accuracy values
acc_values = history.history['accuracy']
val_acc_values = history.history['val_accuracy']
loss_values = history.history['loss']
val_loss_values = history.history['val_loss']
epochs_range = range(1, len(acc_values) + 1)

plt.figure()
plt.plot(epochs_range, acc_values, label='Training Accuracy')
plt.plot(epochs_range, val_acc_values, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss values
plt.figure()
plt.plot(epochs_range, loss_values, label='Training Loss')
plt.plot(epochs_range, val_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
