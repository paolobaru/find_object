# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:12:12 2020

@author: paolo

"""

import cv2

def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),\
                            -min(0, x1), max(x2 - img.shape[1], 0),cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2

def imcrop(img, bbox):
   x1, y1, x2, y2 = bbox
   # if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
   #      img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
   return img[y1:y2, x1:x2, :]