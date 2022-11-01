from __future__ import print_function
import sys
import cv2

import numpy as np
from random import randint
from imutils.video import FPS

width = 320
confThreshold = 0.5
nmsThreshold = 0.3
fps = FPS().start()


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


def findclosest(position, trackerlist,visiitedlist):
  min_index = 0
  min = 500
  for i in range(len(trackerlist)):
    if(abs(trackerlist[i]-position) < min and (i==0 or trackerlist[i]!=0)):
      min = abs(trackerlist[i]-position)
      min_index=i
  if(min < 40 and not min_index in visiitedlist):
    return min_index
  return -1




# Set video to load
videoPath = "input1.mp4"


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

midpoint_tracker = [0,0]
midpoint_detector = 0


# Specify the tracker type
trackerType = trackerTypes[0]#"CSRT"    

# Create single tracker object
trackers=[]
#tracker = createTrackerByName(trackerType)
#trackers.append(tracker)
# trackers.append(tracker)
# trackers.append(tracker)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

counter = 0

# Process video and track objects
while cap.isOpened():
  
  fps.start()
  success, frame = cap.read()
  counter+=1
  frame_counter +=1
  if(frame_counter==1):
    frame_height,frame_width = frame.shape[:2]
  
    print(frame_height,frame_width)
  
  
  if not success:
    break
  for i in range(len(trackers)):
    # get updated location of objects in subsequent frames
    success, box = trackers[i].update(frame)
    #print(len(boxes),frame_counter)

    if(i==0):
        print(len(trackers),midpoint_tracker,midpoint_detector)
    # draw tracked object
    if success:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h),
          (0, 255, 0), 2)
        midpoint_tracker[i]=x+(w/2)

  
        
 
  
  
  result = detectObject(frame)
  #result = [(231, 185, 300, 370)]
  if(len(result)==0):
    print("no detection")
  elif(len(result)==1):
    print("one detection")
    x,y,w,h = result[0][0],result[0][1],result[0][2],result[0][3]
    visited_closest= []
    midpoint_detector=x+(w-x)/2
    closest_index = findclosest(midpoint_detector,midpoint_tracker,visited_closest)
    print("closest_index = ",closest_index)
    if(closest_index != -1):
      #del(trackers[closest_index])
      visited_closest.append(closest_index)
      print("Updating")
      if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
        #tracker = createTrackerByName(trackerType)
        #trackers.append(tracker)
        trackers[closest_index].init(frame,(x,y,w-x,h-y))
        midpoint_tracker[closest_index]=(x+(w-x)/2)
    else:
      if(len(trackers)<=1):
        print("Appending")
        if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
          tracker = createTrackerByName(trackerType)
          trackers.append(tracker)
          trackers[len(trackers)-1].init(frame,(x,y,w-x,h-y))
          midpoint_tracker[len(trackers)-1]=(x+(w-x)/2)
      else:
        print("clearing") 
        trackers=[trackers[1]]
        if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
          tracker = createTrackerByName(trackerType)
          trackers.append(tracker)
          trackers[1].init(frame,(x,y,w-x,h-y))
          midpoint_tracker[len(trackers)-1]=(x+(w-x)/2)
  elif(len(result)==2):
    print("two detections")
    visited_closest= []
    for i in range (2):
      x,y,w,h = result[i][0],result[i][1],result[i][2],result[i][3]      
      midpoint_detector=x+(w-x)/2
      closest_index = findclosest(midpoint_detector,midpoint_tracker,visited_closest)
      print("closest_index = ",closest_index)
      if(closest_index != -1):
        #del(trackers[closest_index])
        visited_closest.append(closest_index)
        print("Updating")
        if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
          #tracker = createTrackerByName(trackerType)
          #trackers.append(tracker)
          trackers[closest_index].init(frame,(x,y,w-x,h-y))
          midpoint_tracker[closest_index]=(x+(w-x)/2)
      else:
        if(len(trackers)<=1):
          print("Appending")
          if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
            tracker = createTrackerByName(trackerType)
            trackers.append(tracker)
            trackers[len(trackers)-1].init(frame,(x,y,w-x,h-y))
            midpoint_tracker[len(trackers)-1]=(x+(w-x)/2)
        else:
          print("clearing")
          if(w<frame_width and h<frame_height and x<frame_width and y<frame_height and y>0 and w>0 and h>0 and x>0 ):
            trackers[i].init(frame,(x,y,w-x,h-y))
            midpoint_tracker[i]=(x+(w-x)/2)


  #print(midpoint_tracker,midpoint_detector)
  # show frame
  cv2.imshow('MultiTracker', frame)
  fps.stop()
  print(1/fps.elapsed())
  frame = cv2.resize(frame, (640, 480),interpolation= cv2.INTER_LINEAR)
  out.write(frame.astype('uint8'))

  
  
  
  # quit on ESC button
  if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
    break

# and release the output
out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
  
  
