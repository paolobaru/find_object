# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 18:50:19 2020

@author: paolo
"""
# import the necessary packages
# from tensorflow.keras.applications import ResNet50
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
from  modules.det_tool import map_results
from simple_detector import detect_simple
import matplotlib.pyplot as plt
import os

class_info = [ { "label" : "squirrel", "folder" : 'C:/projects/find_object/dataset/train/squirrels' } ]

for this_class in class_info:
    
    ( correct , wrong) = (0,0)
    # dirpath = 'C:/projects/find_object/dataset/train/squirrels/'
    dirpath = this_class["folder"]
    for entry in os.scandir(dirpath):
        
        if (entry.path.endswith(".jpg")  and entry.is_file()) :
            
            detection_result =  detect_simple( os.path.join( dirpath , entry  ), export = True)
            
            for label, image in detection_result:
                if label == this_class["label"]:
                    if not os.path.exists( os.path.join( dirpath , "detection") ) : os.mkdir( os.path.join( dirpath , "detection"))
                    cv2.imwrite(os.path.join( dirpath , 
                                             "detection" , 
                                             os.path.splitext(entry.name)[0] + '_raw_det_' + label + os.path.splitext(entry.name)[1]  ), image)
            # print(".", end = "")
    print(entry.name + " Done")