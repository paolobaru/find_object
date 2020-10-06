# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:01:31 2020

@author: paolo
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


img_width, img_height = 128, 128

model= keras.models.load_model("squirrel_model")



probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predict_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

prediction_generator = predict_datagen.flow_from_directory(
        './dataset/video',
        class_mode=None,
        batch_size = 1,
        # target_size=(img_width, img_height),
        shuffle = False
        )

#%%
# outcome =  probability_model.predict_classes(prediction_generator )
nb_validation_samples  = prediction_generator.samples

prediction_generator.reset()
pred = model.predict( prediction_generator , batch_size=None)

#%%
predicted_class_indices=np.argmax(pred,axis=1)
labels=(prediction_generator.class_indices)
labels2=dict((v,k) for k,v in labels.items())
predictions=[labels2[k] for k in predicted_class_indices]
print(predicted_class_indices)
print(labels)
print(predictions)
