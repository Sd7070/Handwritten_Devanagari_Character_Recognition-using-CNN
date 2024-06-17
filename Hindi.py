import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

trainDataGen = ImageDataGenerator(
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
trainGenerator = trainDataGen.flow_from_directory(
    r"C:\Users\Arshad\OneDrive\Desktop\saniraj\Hindi_Dataset\Hindi_Dataset\Train",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
    r"C:\Users\Arshad\OneDrive\Desktop\saniraj\Hindi_Dataset\Hindi_Dataset\Test",
    target_size=(32, 32),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical')

model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=1,
                 activation="relu",
                 input_shape=(32, 32, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding="same"))

model.add(Conv2D(filters=32,
                 kernel_size=(3, 3),
                 strides=1,
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding="same"))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=1,
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding="same"))

model.add(Conv2D(filters=64,
                 kernel_size=(3, 3),
                 strides=1,
                 activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2),
                       strides=(2, 2),
                       padding="same"))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())

model.add(Dense(46, activation="softmax"))

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print(model.summary())

from tensorflow.keras.callbacks import ModelCheckpoint

# Define a ModelCheckpoint callback to save the model
checkpoint = ModelCheckpoint("HindiModel2.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Train the model with the ModelCheckpoint callback
history = model.fit(
    trainGenerator,
    epochs=25,
    validation_data=validation_generator,
    callbacks=[checkpoint]  # Pass the checkpoint callback
)

# Plot accuracy vs epoch and loss vs epoch
import matplotlib.pyplot as plt

# Plot accuracy vs epoch
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Plot loss vs epoch
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

