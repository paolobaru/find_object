# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 23:27:26 2020

@author: paolo
"""

import cv2
import os
from pathlib import Path

  
# Function to extract frames 
def FrameCapture(path): 
      
    # Path to video file 
    vidObj = cv2.VideoCapture(path) 
  
    # Used as counter variable 
    count = 0
  
    # checks whether frames were extracted 
    success = 1
  
    while success: 
  
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read() 
        
        (basepath, fileame) = os.path.split(path)
        framespath = os.path.join ( basepath , 'frames' )
  
        if not os.path.exists(framespath):
            os.makedirs(framespath)    
  
        # Saves the frames with frame-count 
        if success : cv2.imwrite( os.path.join ( framespath , "frame%d.jpg" ) % count, image) 
  
        count += 1

# Driver Code 
if __name__ == '__main__': 
  
    # Calling the function 
    FrameCapture(r"C:\Repositories\find_object\dataset\video\Living Room Arlo_20200929_154516.mp4") 