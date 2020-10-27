# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 21:57:57 2020

@author: paolo


"""


import tensorflow as tf
from tensorflow import keras
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import shutil

import json

    
def predict_image_from_path_vgg16 ( img_path , mute=False ):
    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = keras.applications.vgg16.preprocess_input(x)
    found_label_index = [(x[0]) for x in np.argwhere(full_model.predict(x)[0] > 0.50)]
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
full_model=keras.models.load_model("srhcs_model_vgg16")
data_classes = ('squirrels','racoon','hedgehog','cat','skunk')

#%%
# predict_image_from_path_vgg16 ( 'C:/projects/find_object/dataset/train/squirrels/squirrels1.jpg' )
class_info = [ { "label" : "cat", "folder" : 'C:/Users/paolo/OneDrive/Pictures/scraping/cat_profile' } ]
              
for this_class in class_info:
    
    ( correct , wrong) = (0,0)
    # dirpath = 'C:/projects/find_object/dataset/train/squirrels/'
    dirpath = this_class["folder"]
    for entry in os.scandir(dirpath):
        
        if (entry.path.endswith(".jpg")  and entry.is_file()) :
            labels = predict_image_from_path_vgg16 ( os.path.join( dirpath , entry  ) , mute = True)
            if this_class["label"] not in labels:
                print ( os.path.join( dirpath , entry  ) + " misclassfied as " + str(labels))            
                # predict_image_from_path_vgg16 ( dirpath + entry.name , mute = False)
                # img=image.load_img(dirpath + entry.name)
                # plt.imshow(img)
                plt.figure()
                plt.imshow(plt.imread(os.path.join( dirpath , entry  )))
                plt.text(0.5, 0.5, " misclassfied as " + str(labels), horizontalalignment='center', verticalalignment='center')
                    
                wrong += 1
                
            else:
                correct += 1
                if not os.path.exists( os.path.join( dirpath , "selected") ) : os.mkdir( os.path.join( dirpath , "selected"))
                shutil.move( os.path.join( dirpath , entry.name  ), os.path.join( dirpath , "selected" , entry.name  ) )
    
    print( this_class["label"] + " has " + str(correct) + " correct detectio and " + str(wrong) + " wrong detection")

