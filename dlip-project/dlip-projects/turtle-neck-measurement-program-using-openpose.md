# Turtle neck measurement program using OpenPose

**Date: 2022-06-19** **Author:** Inwoong Kim(21700150), Hyukjun Ha(21700763) **Github:** https://github.com/okdlsdnd/DLIP\_Final/blob/main/Final\_Lab.md\
**Demo video**: https://youtu.be/jZRFZqah7\_A



## Introduction

Because of increasing usage of smartphones and computers, forward head posture is severe in modern young people. But actually by correcting their posture, it can be easily cured. And if you know your current posture, it is naturally followed by correcting posture. So we designed this program to visualize subject's posture by using OpenPose. OpenPose works best at Nvidia GPU environment.



### 1. Download OpenPose

Download OpenPose ZIP file below.(openpose-master.zip)

https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation

![downloadzip](https://user-images.githubusercontent.com/80805040/174346397-a53052f8-1b36-4fd7-8e3a-60c29dee9061.png)

Extract ZIP file in your directory. And start 'getModels.bat' in your directory. ![getmodel](https://user-images.githubusercontent.com/80805040/174346932-2a96aaf9-5af9-4f12-8e0b-829e09a68685.png)

After download is done, you'll have .caffemodel file in each model folder. Copy and paste both .prototxt and .caffemodel files in your directory. ![model](https://user-images.githubusercontent.com/80805040/174420516-e1eea931-bb69-458e-86c9-2bf4da39f711.png)

After this your workplace will be look like this. ![directory](https://user-images.githubusercontent.com/80805040/174420782-ef20bdb1-cc11-437f-ad0a-d4e6be3ff0d2.png)

### 2. VS Code

You need py39, you must check this.

**Module Import**

```
import cv2
import math
import numpy as np
```

**Load OpenPose Model**

```
BODY_PARTS = { "Neck": 1, "Waist": 8, "Left Ear": 17, "Right Ear": 18, "Background": 25 }

POSE_PAIRS = [["Left Ear", "Neck"], ["Right Ear", "Neck"], ["Neck", "Waist"]]

protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_584000.caffemodel"

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
```

**Visualize the Results**

```
# Image Capture from Webcam
cap = cv2.VideoCapture(1)

while True : 
    # Read Images
    ret, image = cap.read()

    imageCopy = image

    # Resize for Performance
    imageCopy = cv2.resize(imageCopy, (1200,680))
    image = cv2.resize(image,(300,170))

    # Get Results from Model
    imageHeight, imageWidth, _ = image.shape
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3])

    points = []

	# Get Coordinates of the Results
    for i in range(0,25):
        probMap = output[0, i, :, :]
    
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H
  
        if prob > 0.1 :    
            points.append((int(x), int(y)))
        else :
            points.append(None)

        if i == 1 :
            neckx = int(x)
            necky = int(y)
        
        if i == 8 :
            waistx = int(x)
            waisty = int(y)

        if i == 17 :
            headx = int(x)
            heady = int(y)
        
        if i == 18 and points[17] is None :
            headx = int(x)
            heady = int(y)
	
	# Calculate the Degree(abs for every direction)
    deg = abs(abs(math.atan((necky-waisty)/((neckx-waistx)+1.e-09))) - abs(math.atan((heady-necky)/((headx-neckx)+1.e-09))))*180/np.pi
    
    # Draw Gauges and Text for Visualizing
    imageCopy =  cv2.rectangle(imageCopy, (0, 0), (350, 50), (255, 255, 255), -1)

    text = str('lean back %s [deg]' % (int(deg)))

    if deg <= 30 :
        ratio = deg/30
        bias = 1
    elif deg > 30 :
        ratio = 1
        bias = 0

    cv2.putText(imageCopy, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(255*bias), int(255*ratio)), 2)
    cv2.rectangle(imageCopy, (1100, 10), (1180, 670), (255-ratio*255, 255-ratio*255, 225 + 30*ratio), 1)
    cv2.rectangle(imageCopy, (1100, int(670-ratio*660)), (1180, 670), (255-ratio*255, 255-ratio*255, 225 + 30*ratio), -1)
    
    # Get Coordinates for Each Pose Pair
    for pair in POSE_PAIRS:
        partA = pair[0]            
        partA = BODY_PARTS[partA]   
        partB = pair[1]             
        partB = BODY_PARTS[partB]   
        
        # Draw Lines for Each Pairs
        if points[partA] and points[partB]:
            x1 = points[partA][0] * 4
            y1 = points[partA][1] * 4
            x2 = points[partB][0] * 4
            y2 = points[partB][1] * 4
            cv2.line(image, points[partA], points[partB], (0, int(255*bias), int(255*ratio)), 2)
            cv2.line(imageCopy, (x1, y1), (x2, y2), (0, int(255*bias), int(255*ratio)), 2)

	# Show Results
    cv2.imshow("result",imageCopy)
    if cv2.waitKey(1) == ord('q'):
        break
```

```
cap.release()
cv2.destroyAllWindows()
```



### Result

| (a) Trutle Neck   | ![turtle](https://user-images.githubusercontent.com/80805040/174421481-86f0d07b-7252-488c-bffa-c6f69801f69e.png)       |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------- |
| (b) Straight Neck | ![straight](https://user-images.githubusercontent.com/80805040/174421475-5905e6f8-7b83-4eb9-8900-c0fe7d15c501.png)     |
| (c) Near Turtle   | ![near\_danger](https://user-images.githubusercontent.com/80805040/174421478-c0055e90-3569-4617-86a2-0e90834ba694.png) |
| (d) Lean Back     | ![lean\_back](https://user-images.githubusercontent.com/80805040/174424208-a3594e97-7f4e-415d-8e53-35ef066fdd7e.png)   |

### Discussion

OpenPose is very heavy program, so it works very slow. Due to the low fps, output shows slightly obsolete results. And OpenPose requires the position of waist, the camera must be set in exact height and distant. The distance is about 1.5m and the height is about 1m, which will lead to inconvenience for use. We need to improve these problems.

### Appendix

**코드**

```
import cv2
import math
from matplotlib.pyplot import hsv
import numpy as np

BODY_PARTS = { "Neck": 1, "Waist": 8, "Left Ear": 17, "Right Ear": 18, "Background": 25 }

POSE_PAIRS = [["Left Ear", "Neck"], ["Right Ear", "Neck"], ["Neck", "Waist"]]
    
protoFile = "pose_deploy.prototxt"
weightsFile = "pose_iter_584000.caffemodel"
 
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

cap = cv2.VideoCapture(1)

while True : 
    ret, image = cap.read()

    imageCopy = image

    imageCopy = cv2.resize(imageCopy, (1200,680))

    image = cv2.resize(image,(300,170))

    imageHeight, imageWidth, _ = image.shape
    
    inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
    
    net.setInput(inpBlob)

    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    print("이미지 ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3])

    points = []

    for i in range(0,25):
        probMap = output[0, i, :, :]
    
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        x = (imageWidth * point[0]) / W
        y = (imageHeight * point[1]) / H
   
        if prob > 0.1 :    
            points.append((int(x), int(y)))
        else :
            points.append(None)

        if i == 1 :
            neckx = int(x)
            necky = int(y)
        
        if i == 8 :
            waistx = int(x)
            waisty = int(y)

        if i == 17 :
            headx = int(x)
            heady = int(y)
        
        if i == 18 and points[17] is None :
            headx = int(x)
            heady = int(y)

    deg = abs(abs(math.atan((necky-waisty)/((neckx-waistx)+1.e-09))) - abs(math.atan((heady-necky)/((headx-neckx)+1.e-09))))*180/np.pi
    
    imageCopy =  cv2.rectangle(imageCopy, (0, 0), (350, 50), (255, 255, 255), -1)

    text = str('lean back %s [deg]' % (int(deg)))

    if deg <= 30 :
        ratio = deg/30
        bias = 1
    elif deg > 30 :
        ratio = 1
        bias = 0

    cv2.putText(imageCopy, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, int(255*bias), int(255*ratio)), 2)
    cv2.rectangle(imageCopy, (1100, 10), (1180, 670), (255-ratio*255, 255-ratio*255, 225 + 30*ratio), 1)
    cv2.rectangle(imageCopy, (1100, int(670-ratio*660)), (1180, 670), (255-ratio*255, 255-ratio*255, 225 + 30*ratio), -1)
    
    for pair in POSE_PAIRS:
        partA = pair[0]            
        partA = BODY_PARTS[partA]   
        partB = pair[1]             
        partB = BODY_PARTS[partB]   
        
        if points[partA] and points[partB]:
            x1 = points[partA][0] * 4
            y1 = points[partA][1] * 4
            x2 = points[partB][0] * 4
            y2 = points[partB][1] * 4
            cv2.line(image, points[partA], points[partB], (0, int(255*bias), int(255*ratio)), 2)
            cv2.line(imageCopy, (x1, y1), (x2, y2), (0, int(255*bias), int(255*ratio)), 2)

    cv2.imshow("result",imageCopy)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

**Flow Chart**

![flow\_chart](https://user-images.githubusercontent.com/80805040/174468464-3c8d5ff5-0ece-4e69-86d3-69749709f322.png)
