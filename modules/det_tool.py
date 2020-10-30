# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:38:50 2020

@author: paolo
"""

import numpy as np
import argparse
import imutils
import time
import cv2

def get_main_box ( prob_map, threshold = 0.10 ):
    qualified_area = np.argwhere( prob_map > prob_map.max() * threshold )
    xx_min = min(x[0] for x in qualified_area )
    yy_min = min(x[1] for x in qualified_area )
    xx_max = max(x[0] for x in qualified_area )
    yy_max = max(x[1] for x in qualified_area )
    
    return xx_min , yy_min, xx_max , yy_max

def map_results (boxes, proba, image):
    y, x, z = image[:,:].shape
    #init probs map
    prob_map = np.zeros((x, y), dtype=np.float)
    
    for i, box in enumerate(boxes):
        xx1, yy1 , xx2, yy2 = box
        box_map = np.ones(( abs(xx2 - xx1 + 1) , abs(yy2 - yy1 + 1)), dtype=np.float) * proba[i]
        prob_map[xx1:xx2+1, yy1:yy2+1] = np.add (prob_map[xx1:xx2+1, yy1:yy2+1] , box_map)
    
    # cv2.imshow("prob maps", prob_map.transpsoe())
    
    xx_min , yy_min, xx_max , yy_max = get_main_box(prob_map, threshold = 0.02)

    
    return xx_min , yy_min, xx_max , yy_max