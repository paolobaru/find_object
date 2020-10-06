# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:24:37 2020

@author: paolo
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

#%%

#train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#      '/tf/tensorflow_squirrels/data',
#    labels='inferred', 
#    label_mode = 'int',
#    image_size = (64,64),
#      color_mode="grayscale")

img_width, img_height = 128, 128

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  
        validation_split=0.2,   
        shear_range=0.15,
        zoom_range=0.15,
        brightness_range=(0.4, 1.0),
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True
        )
train_generator = train_datagen.flow_from_directory(
        './dataset/train',
        target_size=(img_width, img_height),
        subset="training",
        # color_mode="grayscale",
        batch_size=32, 
        class_mode="sparse" 
        )
        # )
validation_generator = train_datagen.flow_from_directory(
        './dataset/train',
        target_size=(img_width, img_height),
        subset="validation",
        # color_mode="grayscale",
        batch_size=32, 
        class_mode="sparse" 
        )

from keras import backend as K 


label_map = (train_generator.class_indices)
index_to_label = dict(zip(label_map.values(), label_map.keys()))

# if K.image_data_format() == 'channels_first': 
#     input_shape = (1, img_width, img_height) 
# else: 
input_shape = (img_width, img_height, 3)
#%%

model = keras.Sequential([
  keras.layers.Conv2D(16, 3, padding='same', activation='relu',input_shape = input_shape),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
  keras.layers.MaxPooling2D(),
  keras.layers.Flatten(),
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(3), #number of classes  
  keras.layers.Softmax()
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


#%%

model.fit(train_generator, 
        epochs=75,
        validation_data=validation_generator,
        verbose =1)
        #validation_data=validation_generator,
        #validation_steps = 4)
        
#%%

model.save("squirrel_model")

#%%

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# validation_generator = test_datagen.flow_from_directory(
#         './dataset/train',
#         color_mode="grayscale",
#         target_size=(64,64),
#         batch_size=4,
#         class_mode='sparse')

test_loss, test_acc = model.evaluate(validation_generator , verbose=2)

print('\nTest accuracy:', test_acc)

#%%

