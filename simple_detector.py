# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:23:54 2020

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
import os, contextlib

import matplotlib.pyplot as plt

# import the necessary packages
import imutils
def sliding_window(image, step, ws):
    # slide a window across the image
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield the current window
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])
            
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image

def decode_predictions(preds):
    top = 1
    # data_classes = ('squirrels','racoon','hedgehog','cat','skunk')
    CLASS_INDEX = {"0": ["squirrel", "squirrel"], 
                   "1": ["raccoon", "raccoon"], 
                   "2": ["hedgehog", "hedgehog"], 
                   "3": ["cat", "cat"], 
                   "4": ["skunk", "skunk"]
                   }
            
    results = []       
    for pred in preds:
      top_indices = pred.argsort()[-top:][::-1]
      result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
      result.sort(key=lambda x: x[2], reverse=True)
      results.append(result)
    return results

    
def detect_simple ( path , size = (40, 40) , min_conf = 0.98, visualize = -1, export = False  ):
    
    # initialize variables used for the object detection procedure
    WIDTH = 600
    PYR_SCALE = 1.5
    WIN_STEP = 8
    ROI_SIZE = size
    INPUT_SIZE = (224, 224)
    
    # load our network weights from disk
    print("[INFO] loading network...")
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            model = keras.models.load_model("srhcs_model_vgg16",    )

    # load the input image from disk, resize it such that it has the
    # has the supplied width, and then grab its dimensions
    orig = cv2.imread(path)
    orig = imutils.resize(orig, width=WIDTH)
    (H, W) = orig.shape[:2]
    
    # initialize the image pyramid
    pyramid = image_pyramid(orig, scale=PYR_SCALE, minSize=ROI_SIZE)
    # initialize two lists, one to hold the ROIs generated from the image
    # pyramid and sliding window, and another list used to store the
    # (x, y)-coordinates of where the ROI was in the original image
    rois = []
    locs = []
    # time how long it takes to loop over the image pyramid layers and
    # sliding window locations
    start = time.time()
    
    # loop over the image pyramid
    for image in pyramid:
        # determine the scale factor between the *original* image
        # dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])
        # for each layer of the image pyramid, loop over the sliding
        # window locations
        for (x, y, roiOrig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # scale the (x, y)-coordinates of the ROI with respect to the
            # *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # take the ROI and preprocess it so we can later classify
            # the region using Keras/TensorFlow
            roi = cv2.resize(roiOrig, INPUT_SIZE)
            roi = img_to_array(roi)
            roi = preprocess_input(roi)
            # update our list of ROIs and associated coordinates
            rois.append(roi)
            locs.append((x, y, x + w, y + h))
            
            # check to see if we are visualizing each of the sliding
            # windows in the image pyramid
            if visualize > 0:
                # clone the original image and then draw a bounding box
                # surrounding the current region
                clone = orig.copy()
                cv2.rectangle(clone, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
                # show the visualization and current ROI
                cv2.imshow("Visualization", clone)
                cv2.imshow("ROI", roiOrig)
                cv2.waitKey(0)
                
    # show how long it took to loop over the image pyramid layers and
    # sliding window locations
    end = time.time()
    print("[INFO] looping over pyramid/windows took {:.5f} seconds".format(
        end - start))
    # convert the ROIs to a NumPy array
    rois = np.array(rois, dtype="float32")
    # classify each of the proposal ROIs using ResNet and then show how
    # long the classifications took
    print("[INFO] classifying ROIs...")
    start = time.time()
    
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull):
            preds = model.predict(rois, use_multiprocessing=True, workers = 16, verbose = 0)

    end = time.time()
    print("[INFO] classifying ROIs took {:.5f} seconds".format(
        end - start))
    # decode the predictions and initialize a dictionary which maps class
    # labels (keys) to any ROIs associated with that label (values)
    # preds = imagenet_utils.decode_predictions(preds, top=1)
    preds = decode_predictions(preds)
    labels = {}
    # loop over the predictions
    for (i, p) in enumerate(preds):
        # grab the prediction information for the current ROI
        (imagenetID, label, prob) = p[0]
        # filter out weak detections by ensuring the predicted probability
        # is greater than the minimum probability
        if prob >= min_conf:
            # grab the bounding box associated with the prediction and
            # convert the coordinates
            box = locs[i]
            # grab the list of predictions for the label and add the
            # bounding box and probability to the list
            L = labels.get(label, [])
            L.append((box, prob))
            labels[label] = L
    #%%
    #select label with most boxes
    square_counts= ([len(labels[item]) for item in labels])
    max_squares =  square_counts.index(max(square_counts))
    
    #%%
    # loop over the labels for each of detected objects in the image
    return_image = []
    for label in labels.keys():
    # label = list(labels)[max_squares]
        # clone the original image so that we can draw on it
        print("[INFO] showing results for '{}'".format(label))
        clone = orig.copy()
        # loop over all bounding boxes for the current label
        for (box, prob) in labels[label]:
            # draw the bounding box on the image
            (startX, startY, endX, endY) = box
            if visualize >-1:
                cv2.rectangle(clone, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
            
            
        # show the results *before* applying non-maxima suppression, then
        # clone the image again so we can display the results *after*
        # applying non-maxima suppression
        if visualize >-1:cv2.imshow("Before", clone)
        clone = orig.copy()
        
        # extract the bounding boxes and associated prediction
        # probabilities, then apply non-maxima suppression
        boxes = np.array([p[0] for p in labels[label]])
        proba = np.array([p[1] for p in labels[label]])
        
        xx_min , yy_min, xx_max , yy_max =  map_results (boxes, proba,clone, threshold = 0.10)
        
        # boxes = non_max_suppression(boxes, proba, overlapThresh=0.15)
        # loop over all bounding boxes that were kept after applying
        # non-maxima suppression
        # boxes = [(startX, startY, endX, endY)]
        boxes = [(xx_min, yy_min, xx_max, yy_max)]
        for (startX, startY, endX, endY) in boxes:
            # draw the bounding box and label on the image
            cv2.rectangle(clone, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(clone, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
        # show the output after apply non-maxima suppression
        if visualize >-1 : cv2.imshow("After", clone)
        if visualize >-1: cv2.waitKey(0)
        
        if export: return_image.append( (label, clone.copy() ) )
        # input("Press Enter key...")
      
    cv2.destroyAllWindows()
    if export: return   return_image

if __name__ == "__main__":
    
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", 
                    required=True, 
                    default = "C:/projects/find_object/dataset/train/squirrels/squirrel profile100.jpg", 
                    help="path to the input image")
    ap.add_argument("-s", "--size", 
                    type=str, 
                    default="(200, 200)",
        help="ROI size (in pixels)")
    ap.add_argument("-c", "--min-conf", type=float, default=0.95,
        help="minimum probability to filter weak detections")
    ap.add_argument("-v", "--visualize", type=int, default=-1,
        help="whether or not to show extra visualizations for debugging")
    args = vars(ap.parse_args())
    
    
    
    detect_simple ( args["image"],  
                   size = eval(args["size"]) , 
                   visualize=args["visualize"] , 
                   min_conf = args["min_conf"] )