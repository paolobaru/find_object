# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 21:24:37 2020

@author: paolo
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import json

print(tf.__version__)

batch_size = 30
data_classes = ('squirrels','racoon','hedgehog','cat','skunk')

def generators(shape, preprocessing): 
    '''Create the training and validation datasets for 
    a given image shape.
    '''
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocessing,
        horizontal_flip = True, 
        validation_split = 0.1,
        shear_range=0.15,
        zoom_range=0.15,
        brightness_range=(0.4, 1.0),
        width_shift_range=0.25,
        height_shift_range=0.25,
    )

    height, width = shape

    train_dataset = imgdatagen.flow_from_directory(
        './dataset/train',
        target_size = (height, width), 
        classes = data_classes,
        batch_size = batch_size,
        subset = 'training', 
    )

    val_dataset = imgdatagen.flow_from_directory(
        './dataset/train',
        target_size = (height, width), 
        classes = data_classes,
        batch_size = batch_size,
        subset = 'validation'
    )
    return train_dataset, val_dataset

def plot_history(history, yrange):
    '''Plot loss and accuracy as a function of the epoch,
    for the training and validation datasets.
    '''
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Get number of epochs
    epochs = range(len(acc))

    # Plot training and validation accuracy per epoch
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    
    # Plot training and validation loss per epoch
    plt.figure()

    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    
    plt.show()
    
def predict_image_from_path_vgg16 ( img_path , mute=False ):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    found_label_index = [(x[0]) for x in np.argwhere(full_model.predict(x)[0] > 0.60)]
    if not mute: print ( img_path + " ")
    # prediction = full_model.predict(x)
    found_labels = []
    for lb_i in found_label_index:
        found_labels.append( data_classes[lb_i] )
        if not mute: print (found_labels[-1])
    # print()    
    if not mute: plt.imshow(img)
    return found_labels
#%%

# #train_ds = tf.keras.preprocessing.image_dataset_from_directory(
# #      '/tf/tensorflow_squirrels/data',
# #    labels='inferred', 
# #    label_mode = 'int',
# #    image_size = (64,64),
# #      color_mode="grayscale")

# img_width, img_height = 128, 128

# train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255,  
#         validation_split=0.2,   
#         shear_range=0.15,
#         zoom_range=0.15,
#         brightness_range=(0.4, 1.0),
#         width_shift_range=0.25,
#         height_shift_range=0.25,
#         horizontal_flip=True
#         )
# train_generator = train_datagen.flow_from_directory(
#         './dataset/train',
#         target_size=(img_width, img_height),
#         subset="training",
#         # color_mode="grayscale",
#         batch_size=32, 
#         class_mode="sparse" 
#         )
#         # )
# validation_generator = train_datagen.flow_from_directory(
#         './dataset/train',
#         target_size=(img_width, img_height),
#         subset="validation",
#         # color_mode="grayscale",
#         batch_size=32, 
#         class_mode="sparse" 
#         )

# from keras import backend as K 


# label_map = (train_generator.class_indices)
# index_to_label = dict(zip(label_map.values(), label_map.keys()))

# # if K.image_data_format() == 'channels_first': 
# #     input_shape = (1, img_width, img_height) 
# # else: 
# input_shape = (img_width, img_height, 3)
# #%%

# model = keras.Sequential([
#   keras.layers.Conv2D(16, 3, padding='same', activation='relu',input_shape = input_shape),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
#   keras.layers.MaxPooling2D(),
#   keras.layers.Flatten(),
#   keras.layers.Dense(128, activation='relu'),
#   keras.layers.Dense(3), #number of classes  
#   keras.layers.Softmax()
# ])
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


#%%
vgg16 = keras.applications.vgg16


conv_model = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# conv_model.summary()

# flatten the output of the convolutional part: 
x = keras.layers.Flatten()(conv_model.output)
# three hidden layers
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dense(100, activation='relu')(x)
x = keras.layers.Dropout(0.5)(x)
# final softmax layer with 3 categories (dog and cat)
predictions = keras.layers.Dense(len(data_classes), activation='softmax')(x)

# creating the full model:
full_model = keras.models.Model(inputs=conv_model.input, outputs=predictions)

for layer in conv_model.layers:
    layer.trainable = False
    
full_model.summary()

full_model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adamax(lr=0.001),
                  metrics=['acc'])
#%%

train_dataset, val_dataset = generators((224,224), preprocessing=vgg16.preprocess_input)
history = full_model.fit_generator(
    train_dataset, 
    validation_data = val_dataset,
    workers=8,
    epochs=12,
)
plot_history(history, yrange=(0.9,1))
#%%
full_model.save("srhcs_model_vgg16")
#%%
full_model=keras.models.load_model("srhcs_model_vgg16")

#%%
# predict_image_from_path_vgg16 ( 'C:/Repositories/find_object/dataset/train/squirrels/squirrels1.jpg' )
class_info = [ { "label" : "squirrels", "folder" : "C:/Repositories/find_object/dataset/train/squirrels/"} ,
         { "label" : "racoon", "folder"  : "C:/Repositories/find_object/dataset/train/racoon/"},
         { "label" : "hedgehog", "folder"  : "C:/Repositories/find_object/dataset/train/hedgehog/"},
         { "label" : "cat", "folder"  : "C:/Repositories/find_object/dataset/train/cat/"},
         { "label" : "skunk", "folder"  : "C:/Repositories/find_object/dataset/train/skunk/"},
        ]
for this_class in class_info:
    
    ( correct , wrong) = (0,0)
    # dirpath = 'C:/Repositories/find_object/dataset/train/squirrels/'
    dirpath = this_class["folder"]
    for entry in os.scandir(dirpath):
        if (entry.path.endswith(".jpg")  and entry.is_file()) :
            labels = predict_image_from_path_vgg16 ( dirpath + entry.name , mute = True)
            if this_class["label"] not in labels:
                print ( dirpath + entry.name + " misclassfied as " + str(labels))            
                # predict_image_from_path_vgg16 ( dirpath + entry.name , mute = False)
                # img=image.load_img(dirpath + entry.name)
                # plt.imshow(img)
                plt.figure()
                plt.imshow(plt.imread(dirpath + entry.name))
                plt.text(0.5, 0.5, " misclassfied as " + str(labels), horizontalalignment='center', verticalalignment='center')
                    
                wrong += 1
            else:
                correct += 1
    print( this_class["label"] + " has " + str(correct) + " correct detectio and " + str(wrong) + " wrong detection")
        
        
        
# #%%        
# predict_image_from_path_vgg16 ( 'C:/Repositories/find_object/dataset/train/racoon/racoon1.jpg' )
# predict_image_from_path_vgg16 ( 'C:/Repositories/find_object/dataset/train/hedgehog/hedgehog1.jpg' )



# #%%
# model.fit(train_generator, 
#         epochs=75,
#         validation_data=validation_generator,
#         verbose =1)
#         #validation_data=validation_generator,
#         #validation_steps = 4)
        
# #%%


# #%%

# test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# # validation_generator = test_datagen.flow_from_directory(
# #         './dataset/train',
# #         color_mode="grayscale",
# #         target_size=(64,64),
# #         batch_size=4,
# #         class_mode='sparse')

# test_loss, test_acc = model.evaluate(validation_generator , verbose=2)

# print('\nTest accuracy:', test_acc)

# #%%

# label_to_index = train_generator.class_indices
# index_to_label = inv_map = {v: k for k, v in label_to_index.items()}

# with open('squirrel_model_labels.json', 'w') as outfile:
#     json.dump(train_generator.class_indices, outfile)