import cv2
import numpy as np

# Set video to load
videoPath = "input2.mp4"

cap = cv2.VideoCapture(videoPath)
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
#print(classNames)
#print(len(classNames))

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
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                    (x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,0),2)

    

while True:
    success, img = cap.read()

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

    findObjects(outputs,img)

    
    

    cv2.imshow("Image",img)
    cv2.waitKey(1)
    

