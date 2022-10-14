from __future__ import print_function
import sys
import cv2

import numpy as np
from random import randint

width = 320
confThreshold = 0.5
nmsThreshold = 0.3


classesFile = 'coco.names'
classNames = []

modelConfiguration = 'yolov3-tiny.cfg'
modelWeights = 'yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

with open(classesFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
frame_counter = 0

def createTrackerByName(trackerType):
  # Create a tracker based on tracker name
  if trackerType == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackerType == trackerTypes[1]:
    tracker = cv2.TrackerMIL_create()
  elif trackerType == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackerType == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackerType == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackerType == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackerType == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackerType == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)

  return tracker

def findObjects(outputs, img):
  ht,wt,ct = img.shape
  bbox = []
  classIds = []
  confs = []
  detections = []

  for output in outputs:
    for det in output:
      scores = det[5:]
      classId = np.argmax(scores)
      confidence = scores[classId]
      if confidence > confThreshold:
        w,h = det[2]*wt,det[3]*ht
        x,y = int((det[0]*wt) - w/2) , int((det[1]*ht) - h/2)
        bbox.append([x,y,w,h])
        classIds.append(classId)
        confs.append(float(confidence))
    #print(len(bbox))
  indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nmsThreshold)
    #print(indices)

  for i in indices:
    i = i[0]
    box = bbox[i]
    x,y,w,h= int(box[0]),int(box[1]),int(box[2]),int(box[3])
    detections.append((x,y,x+w,y+h))
  return detections



def detectObject(img):
  blob = cv2.dnn.blobFromImage(img,1/255,(width,width),[0,0,0],1,crop=False)
  net.setInput(blob)

  layerNames = net.getLayerNames()
    #print(layerNames) 
    #print(net.getUnconnectedOutLayers())
  outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    # print(outputNames)

  outputs = net.forward(outputNames)
    #print(len(outputs))
    # print(outputs[0].shape)
    # print(outputs[1].shape)

    # print(outputs[0][0])

  detections = findObjects(outputs,img)
  return detections






# Set video to load
videoPath = "input2.mp4"


# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
  print('Failed to read video')
  sys.exit(1)
## Select boxes
bboxes = []
colors = [] 


# Specify the tracker type
trackerType = trackerTypes[0]#"CSRT"    

# Create MultiTracker object
multiTracker = cv2.MultiTracker_create()




# Process video and track objects
while cap.isOpened():
  success, frame = cap.read()
  frame_counter +=1
  
  
  if not success:
    break

  # get updated location of objects in subsequent frames
  success, boxes = multiTracker.update(frame)
  #print(len(boxes),frame_counter)

  # draw tracked objects
  for i, newbox in enumerate(boxes):
    p1 = (int(newbox[0]), int(newbox[1]))
    p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
    if()
    cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

  # show frame
  cv2.imshow('MultiTracker', frame)
  if(frame_counter%60==0):
    #bbox2=(231, 185, 73, 188)
    result = detectObject(frame)
    if(len(result)>0):
      print(result[0],'detecting objects',len(result))
      x,y,w,h = result[0][0],result[0][1],result[0][2],result[0][3]
      cv2.rectangle(frame,(x,y),(w,h),(255,255,0),2)
      colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
      #del multiTracker
      #multiTracker = cv2.MultiTracker_create()
      multiTracker.add(createTrackerByName(trackerType), frame, (x,y,w-x,h-y))

  #cv2.imshow("Image",frame)
  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break
