# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 20:35:40 2020

@author: paolo
"""

# import the necessary packages
from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import datetime
import cv2
import numpy as np
import json
from utils import video_utils
import copy

class movements_log :
    def __init__(self):
        self.log=[]   
        
    def add(self):
        log_entry = { 'event_time' : datetime.datetime.now() }
        self.log.append(log_entry)
        
    def last(self):
        if ( len(self.log)!= 0):
            return self.log[-1]['event_time']
        else: return None
    
    def is_recent(self):
        if self.last() is None: return False
        else :
            delta = (datetime.datetime.now() - self.last()).seconds
            if delta > 5 : return False
            else: return True
    
    def getcount(self):
        return len (self.log)
    
def save_frame_to_file (framecolor_info, gray = False):
    frames = [ f['frame'] for f in framecolor_info ] 
    if gray : (ht, wt, fcount) = np.shape(frames)
    else: (fcount, ht, wt, colors) = np.shape(frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    timestr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    cvwrite = cv2.VideoWriter('C:/Repositories/record/motion_' +
                              timestr +
                              '.mp4', fourcc, 
                              fps = 10.0,
                              frameSize = (wt,ht)
                              )
    
    for frame in frames:
        cvwrite.write(frame)
    cvwrite.release()
    with open('C:/Repositories/record/motion_' +
                              timestr +
                              '.json', 'w') as outfile:
        all_frames = [ { 'frame_index' : index, 
                        'frame_motion_objects': f['frame_motion_objects']} 
                      for index, f in enumerate(framecolor_info) ]
        video_file_info = {'video_data': all_frames}
        json.dump( video_file_info, outfile)
        outfile.close()

#%%    

settings = { 'bkgn_avg_len' : 8 , 
            'past_buffer_len' :25 , 
            'resolution' : {"width": 1920, 'heigth':1080} ,
            'min_mov_area' : 128 * 3
            }
show_feed = True
  
# params for corner detection 
feature_params = dict( maxCorners = 100, 
                       qualityLevel = 0.3, 
                       minDistance = 7, 
                       blockSize = 7 ) 
  
# Parameters for lucas kanade optical flow 
lk_params = dict( winSize = (15, 15), 
                  maxLevel = 2, 
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                              10, 0.03)) 

#%%
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=196, help="minimum area size")
args = vars(ap.parse_args())
# if the video argument is None, then we are reading from webcam
# if args.get("video", None) is None:
# vs = VideoStream(src=0, resolution=(1280 ,1024) , framerate=10.0).start()
# otherwise, we are reading from a video file
# else:
vs = cv2.VideoCapture(0)
vs.set(3,settings['resolution']['width'])
vs.set(4,settings['resolution']['heigth'])
# vs.set(5,10)

time.sleep(2.0)
# initialize the first frame in the video stream
firstFrame = None


movement_detect = movements_log()
framecolor_info=[]
frameavg =[]
framelogs =[]
still_log=[]
initcount = 0
stream_ready = False
gray_sharp_buffer =[]
  
# Create some random colors 
color = np.random.randint(0, 255, (100, 3)) 
#%%
# loop over the frames of the video
while True:
    # grab the current frame and initialize the occupied/unoccupied
    # text
    ready, frame = vs.read()
    # frame = frame if args.get("video", None) is None else frame[1]
    # text = "Unoccupied"
    still_frame = True
    still_log.append(True)
    
    # if the frame could not be grabbed, then we have reached the end
    # of the video
    if frame is None:
        break
    
    # resize the frame, convert it to grayscale, and blur it
    # frame = video_utils.imcrop( frame , ( 40, 0, 1920 - 40, 1080))    
    frame_color = copy.deepcopy(frame)
    framecolor_info.append({'frame' : frame_color, "frame_motion_objects" : []})
    # frame = imutils.resize(frame, width=1280)
    gray_frame_sharp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray_frame = cv2.GaussianBlur(gray_frame_sharp, (21, 21), 0)
    # if the first frame is None, initialize it
    if not stream_ready:
        if firstFrame is None or initcount < 16:
            firstFrame = gray_frame
            initcount += 1
            continue
        elif len(frameavg) < settings['bkgn_avg_len']:
            frameavg.append(gray_frame)
            if (len(frameavg) == settings['bkgn_avg_len']):
                stream_ready = True
                firstFrame = np.average(frameavg, axis=0).astype(dtype=np.uint8)
                p0 = cv2.goodFeaturesToTrack(gray_frame, mask = None, 
                                         **feature_params) 
                  
                # Create a mask image for drawing purposes 
                mask = np.zeros_like(frame) 
                old_gray = gray_frame
                while ( len(gray_sharp_buffer) <  settings['past_buffer_len'] ) : gray_sharp_buffer.append(gray_frame_sharp.copy())
                gray_sharp_buffer.append(gray_frame_sharp)
            continue

        
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray_frame)
    thresh = cv2.threshold(frameDelta, 8, 255, cv2.THRESH_BINARY)[1]
    
    thresh_diff = cv2.threshold(cv2.absdiff(gray_frame, gray_sharp_buffer[-1]), 8, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.dilate(thresh, kernel=np.ones((5,5), np.uint8), iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
          # if the contour is too small, ignore it
          if cv2.contourArea(c) < settings['min_mov_area'] :
              continue
          # compute the bounding box for the contour, draw it on the frame,
          # and update the text
          (x, y, w, h) = cv2.boundingRect(c)
          framecolor_info[-1]["frame_motion_objects"].append( (x, y, w, h) )
          cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    if len(framecolor_info[-1]["frame_motion_objects"]) != 0:
        still_frame = False
        movement_detect.add()
        still_log[-1] = False
              
   #%% 

    if movement_detect.is_recent():
        framecolor_info.append({'frame' : frame_color, "frame_motion_objects" : []})
    else:
        #%% 
        framelogs =   [ f['frame'] for f in framecolor_info ]           
        if len(framelogs) > 1 : save_frame_to_file (framecolor_info)
        framecolor_info.clear() 

  
    
    if show_feed :
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Movement " + str( not still_frame), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (12, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame diff", thresh_diff)
        # cv2.imshow("Optical Flow", cv2.add(frame_flow, mask))
    
    # firstFrame = gray_frame
    if still_frame: 
        frameavg.append(gray_frame.copy())
        frameavg.append(gray_frame.copy())
        frameavg.append(gray_frame.copy())
    else:
        frameavg.append(gray_frame.copy())
    while ( len(frameavg) > settings['bkgn_avg_len'] ):
            frameavg.pop(0)
    firstFrame = np.average(frameavg, axis=0).astype(dtype=np.uint8) 
    
    while ( len(gray_sharp_buffer) >  settings['past_buffer_len'] ) : gray_sharp_buffer.pop(0)
    gray_sharp_buffer.append(gray_frame_sharp.copy())
    
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key is pressed, break from the lop
    

    # old_gray = gray_frame_sharp.copy()
    # p0 = good_new.reshape(-1, 1, 2) 
    if key == ord("q"):
        break
    
# cleanup the camera and close any open windows
vs.release()
cv2.destroyAllWindows()
print ( "recorded " + str(movement_detect.getcount()) + " events")
#%%
# (fcount, ht, wt, colors) = np.shape(framelogs)
if len(framecolor_info) > 1:
    save_frame_to_file (framecolor_info)
    print( "wrote " + str(len(framecolor_info)) + " frame")
    # framelogs.clear()