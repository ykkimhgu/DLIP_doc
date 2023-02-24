# People Counting Embedded System

Date: June,19,2022

Author: Sungjoo Lee, Joseph Shin

Github: https://github.com/SungJooo/DLIP\_FINAL

Demo Video: https://youtu.be/8opxq6PDLP0

## Introduction

This tutorial is about how to create an independent embedded system that can:

1. Film people entering and exiting the room from the top of a doorway
2. Track the people using YOLOv5 detection and user-written code
3. Record how many people entered or exited the room, all within a raspberry pi

## Requirement

### Hardware

* Raspberry Pi 4 Model B 4GB & microSD card 32GB
* Logitech WebCam C920
* Coral TPU USB Accelerator
* Bread-board & LED

### Software Installation

#### Software to test the system on computer

**1. Install libs**

Open Anaconda Prompt and enter the following commands:

* update conda

```python
conda update -n base -c defaults conda
```

* create py39 virtual environment

```python
conda create -n py39 python=3.9.12
```

* activate py39 virtual environment

```python
conda activate py39
```

* install opencv

```python
conda install -c anaconda seaborn jupyter
pip install opencv-python
```

* install pytorch

```python
# CPU Only
conda install -c anaconda seaborn jupyter
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cpuonly -c pytorch
pip install opencv-python torchsummary
```

* install numpy, matplot

```python
conda install numpy
conda install -c conda-forge matplotlib
```

**2. Install YOLOv5**

Access YOLOv5 github then download the following repository

![YOLO1](https://user-images.githubusercontent.com/23421059/169227977-bf94857e-3e87-4cc5-9d1d-daf73836a3dd.png)

Unzip the folder, changing the name from yolov5-master -> yolov5 then paste it to your desired location.

Go in the folder then copy the directory.

![YOLO2](https://user-images.githubusercontent.com/23421059/169229474-723ba3ae-2c70-4bcf-8d4d-760543c79fb1.png)

Open Anaconda Prompt in user mode then enter the following commands.

```python
conda activate py39
cd $YOLOv5PATH$ // [ctrl+V] your copied yolov5 directory
pip install -r requirements.txt
```

![YOLO2](https://user-images.githubusercontent.com/23421059/169230206-55eacf01-0b72-42a2-b8c2-2b046572d5bb.png)

**2. Test the code on VScode**

To download the source video, go the the following URL:

https://github.com/SungJooo/DLIP\_FINAL

There you will find a file called source.MOV. Download it to your desired folder

Now with the software installations complete, open the folder you downloaded the source.MOV file to in VScode, then create a new .py file and paste the "code to test on computer" code, which you can find in the appendix of this paper.

![cap](https://user-images.githubusercontent.com/91940808/174497207-7d6a2e0f-dce3-4abe-8638-f212f5e0429a.png)

Near the top of the code (lines 8\~13) you will see this part. If you wish to use the webcam for your test, use the cap=cv2.VideoCapture(0) line and comment out the above line, and if you wish to use the source video, use the cap=cv2.VideoCapture('source.MOV') line and comment out the below line.

Then, run the code on VScode.

## Software Explanation

The algorithm works in three steps:

1. Object detection via YOLOv5 pretrained model
2. Object tracking using the detection data
3. People counting using the tracking data

The following are some important functions in the tracking or people counting algorithm

### matchboxes()

Uses the coordinates of all bounding boxes in previous frame and current frame to match the boxes with each other and track the objects

```python
def matchboxes(coordlist,prev_coordlist,width):
```

**Parameters**

* **coordlist:** list of bounding box coordinates in current frame
* **coordlist:** list of bounding box coordinates in previous frame
* **width:** width of frame

**Example Code**

```python
# list of boxes that have corresponding boxes in previous frame
i_list=matchboxes(coordlist, prev_coordlist,width)
```

### checkbot\_box()

Checks if the inputted box coordinates are near the bottom of the frame

```python
def checkbot_box(coords,height):
```

**Parameters**

* **coords:** coordinates of box
* **height:** height of frame

**Example Code**

```python
if checkbot_box(new_coords,height)==1:
	num_people-=1
```

### update\_frame()

Updates the frame information using the object detection data and previous number of people

```python
def update_frame(results,prev_results,frame,rect_frame,num_people):
```

**Parameters**

* **results:** YOLOv5 detection current frame results
* **prev\_results:** YOLOv5 detection previous frame results
* **frame:** captured frame of video
* **rectframe:** frame with colored-in bounding boxes
* **num\_people:** number of people from previous frame

**Example Code**

```python
frame,num_people=update_frame(results,prev_results,frame,rect_frame,num_people)
```

## Tutorial Procedure

#### Raspberry Pi Setup

**1. Install Raspberry Pi OS**

If you want to start Raspberry Pi, you need to install the Raspberry Pi OS. First, insert a micro SD card with a reader into your laptop. Then, download the OS installer in the [link](https://www.raspberrypi.com/software/). You can get an exe. file named "imager\_1.7.2.exe". When you run the file, you **must** pick 64-bit OS for activating YOLOv5.

![install\_guide](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/rasp\_software\_05.png)

**1.1. Remote controlling Raspberry Pi**

For remote controlling the Raspberry Pi in laptop, you have to do add "**ssh**" file which doesn't have any extension name, and "**wpa\_supplicant.conf**" file at the root folder of Raspberry Pi.

The file named "wpa\_supplicant.conf" need to include the following:

```
ctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev
update_config=1
country=GB
network={   
	ssid="JooDesk"
	psk="wwjd0000"
}
```

The name "JooDesk" and the password "wwjd0000" is the laptop hotspot's ID and password, to make the IP address unchanged. The bandwidth must be selected as 2.4GHz.

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/rasp\_remote\_04\_new.png)

When you've done the process above, insert mirco SD card to Raspberry Pi and boot it. After you boot the Raspberry Pi, make your hostname and password.

If the Raspberry Pi is executed successfully, you have to make the IP address of Raspberry Pi as a static IP address. To do this, you need to follow the instructions:

```
$ hostname -I	// chech the IP address of Raspberry Pi
$ sudo apt-get install vim	// install vim
$ sudo vi /etc/dhcpcd.conf	// 'i' to insert, 'esc' to out
```

And then, insert the following command at the bottom of it.

```
interface wlan0
static ip_address=192.168.137.110/24
static routers=192.168.137.1
static domain_name_servers=192.168.137.1
```

There should be no "#" marks. After you enter the command, press "esc" to quit, and type ":wq!" to quit the vim. After this step, your Raspberry Pi will get the static IP address, which is "192.168.137.110". Then reboot your Raspberry Pi with the command "$ reboot".

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/staticip.jpg)

After reboot the Raspberry Pi, follow the command for the next step.

```
$ sudo apt-get update	// command for update packages
$ sudo apt-get install tightvncserver
```

Tightvncserver is a program to synchronize the Raspberry Pi screen on a laptop. Specific guidelines are follows.

**1.2. PuTTY**

PuTTY is a program for connecting to Raspberry Pi as a SSH mode. You can download via the [link](https://www.putty.org).

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/rasp\_remote\_01\_new.png)

The static IP address is "192.168.137.110", and use the port number as 22.

**1.3. TightVNC**

When connecting to Raspberry Pi using PuTTY, it is connected only in terminal mode. To connect and use the Raspberry Pi as a GUI environment, TightVNC program would help. The installation link is [here](https://www.tightvnc.com/). If you download the TightVNC, you have to set the password.

By the above step, you have installed TightVNC at the Raspberry Pi. To activate the TightVNC in the Raspberry Pi, command the following line.

```
$ tightvncserver
$ sudo netstat –tulpn	// check the state of Raspberry Pi
```

TightVNC uses the 5901 port of Raspberry Pi. By the command "sudo netstat –tulpn", you can check the state 0.0.0.0.0:5901 is in the "listen" state. If it is, it is ready to sync Raspberry Pi into your laptop. "$ vncpasswd" is a command to edit the password of TightVNC.

**2. YOLOv5 in Raspberry Pi**

First, you have to clone the YOLOv5 repository into the Raspberry Pi. To do this, you need to enter the following command line.

```
$ git clone https://github.com/ultralytics/yolov5
```

After this, the "yolov5" folder would be formed at the root folder of Raspberry Pi. Follow the instruction to make the environment for yolov5.

```
$ cd yolov5
$ pip3 install -r requirements.txt
$ pip3 install numpy --upgrade
$ python detect.py --source data/images --weights yolov5n.pt --conf 0.25
```

After the commands, you can find the following image at the "runs/detect/exp" in Raspberry Pi.

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/zidane.jpg)

For connecting external camera input device(such as Logitech Webcam, Picam ...), you can test the module that can detect the object by the input source. The test code is as follows. "source 0" means the external device you have connected to the Raspberry Pi. For the extra device, "source 1" and goes on.

```
$ python detect.py --source 0 --weights yolov5n.pt --conf 0.25
```

As Raspberry Pi environment isn't good as laptop, the upper limit of using yolo model is yolov5n to prevent of much less FPS.

**2.1. Package RPi.GPIO**

This is the python package to control the GPIO on a Raspberry Pi to turn on the light. To do this, follow the command in Raspberry Pi.

```
$ sudo apt-get upgrade
$ sudo apt-get install rpi-gpio
```

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/rasp\_gpio\_pinmap.png)

This is the pinmap of Raspberry Pi 4 GPIO. We used GPIO21(Pin 40) as a voltage source to the light. To activate this on python, the code is as follows.

```python
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
pin_num = 21
GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)

def GPIO_LIGHT(numPeople, frame):
    if numPeople > 0: GPIO.output(pin_num, GPIO.HIGH)
    else: GPIO.output(pin_num, GPIO.LOW)
```

**3. Algorithm implementation in Raspberry Pi**

To implement the algorithm into the Raspberry Pi, follow the command.

First, upload your code to your github repository. Then, execute the following command at the Raspberry Pi root folder.

```
$ git clone ///your github address///
```

Then, get into your github folder with the command **cd**, and command the following code.

```
$ python ///your .py file///
```

From here, we would guide you how we made it.

```
$ git clone https://github.com/SungJooo/DLIP_FINAL
```

Then you could find the 3 python code which we made.

![](https://raw.githubusercontent.com/SungJooo/DLIP\_FINAL/main/github\_01\_main.png)

"DLIP\_Final\_00\_test.py" is the file that model yolov5n is working well on Raspberry Pi. This file finds only the "person" class.

"DLIP\_Final\_01\_fps.py" is the file that measures your FPS with the model yolov5n. As model is still heavy to covered in Raspberry Pi, the FPS would be about 2\~2.5, and in remote condition, the FPS gets even lower when the WiFi network is bad.

"DLIP\_Final\_01\_fps.py" is the file that turns the lights on and off depending on whether a person enters or leaves.

To launch the code, write down the following code at the DLIP\_FINAL folder.

```
$ python DLIP_Final_10_LAST.py
```

## Results and Analysis

The system was successfully able to:

1. film the doorway entrance
2. use the video footage to detect, track, and count the people in the frame
3. all within a raspberry pi module, without connection to an external computer

Some issues were that the frame rate (around 2.5fps) and accuracy (around 60 percent) of the detection model (YOLOv5 nano) weren't superb on a raspberry pi. YOLOv5 nano was deemed the adequate model, for a lighter model would result in a faster frame rate but a lower accuracy, while a heavier model would've had a higher accuracy, but a higher frame rate.

Below is a table of the frame rate depending on the device.

![fpw](https://user-images.githubusercontent.com/91940808/174496678-e332d19a-855f-4c2f-b4a8-a25a389434bd.png)

A possible solution to this problem to be to use a tensor-based object detection model instead of YOLOv5, which would have increased fps without sacrifice of accuracy. This is because a TPU was used for this project, which is optimized to accelerate computing speed of tensor-based models.

## Appendix

**Code to test on computer**

```python
import torch
import cv2
import random
from PIL import Image
import numpy as np
import math
import time

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes=[0]

# cap = cv2.VideoCapture('source.MOV')
cap = cv2.VideoCapture(0)
width = int(cap.get(3)); height = int(cap.get(4)); 

frameno=0
num_people=0
fpsStart = 0
fps = 0


# returns coordinates of box as list
def box_coords(box):
    xmin=int(box[0])
    ymin=int(box[1])
    xmax=int(box[2])
    ymax=int(box[3])
    return [xmin, ymin, xmax, ymax]

# checks if box touches the bottom of frame
def checkbot_box(coords,height):
    ymax=coords[3]
    if ymax>height-(height/54):
        return 1
    else:
        return 0

# returns center coordinates of box
def box_cent(coords):
    cent_x=int((coords[0]+coords[2])/2)
    cent_y=int((coords[1]+coords[3])/2)
    return [cent_x,cent_y]

# gets intersecting area of two boxes
def inters_area(coord1,coord2):
    xmin1=coord1[0]
    ymin1=coord1[1]
    xmax1=coord1[2]
    ymax1=coord1[3]
    xmin2=coord2[0]
    ymin2=coord2[1]
    xmax2=coord2[2]
    ymax2=coord2[3]
    dx=min(xmax1,xmax2)-max(xmin1,xmin2)
    dy=min(ymax1,ymax2)-max(ymin1,ymin2)
    if (dx>0) and (dy>0):
        return dx*dy
    else:
        return 0

# returns list of coordinates of boxes in current frame that are new (no corresponding box in previous frame)
def newbox(coordlist,i_list):
    new_list=[]
    for k in coordlist:
        if k not in [i[0] for i in i_list]:
            new_list+=[k]
    return new_list

# returns list of coordinates of boxes in previous frame that have disappeared (no corresponding box in current frame)
def dispbox(prev_coordlist,i_list):
    disp_list=[]
    for k in prev_coordlist:
        if k not in [i[1] for i in i_list]:
            disp_list+=[k]
    return disp_list

# finds which box in previous slide is the one in current frame (highest intersecting area)
def matchboxes(coordlist,prev_coordlist,width):
    i_list=[]
    for coord in coordlist:
        area=0
        add_ilist=[]
        for prev_coord in prev_coordlist:
            if inters_area(coord,prev_coord)>area and (math.dist(box_cent(coord),box_cent(prev_coord))<(width/20)):
                area=inters_area(coord,prev_coord)
                add_ilist=[[coord, prev_coord]]
            if coord not in [i[0] for i in i_list] and prev_coord not in [j[1] for j in i_list]:
                i_list+=add_ilist
    return i_list


# COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)
def COUNT_PEOPLE_FRAMEOUT(dataPre, dataCur, frame, frameCopy, num_people):
    # create lists of all box coordinates in previous and current frame
    prev_coordlist=[]
    for j in range(len(dataPre.xyxy[0])):
        prev_coords=box_coords(dataPre.xyxy[0][j])
        prev_coordlist+=[prev_coords]
    coordlist=[]
    for k in range(len(dataCur.xyxy[0])):
        coords=box_coords(dataCur.xyxy[0][k])
        coordlist+=[coords]
    
    for c in coordlist:
        cv2.rectangle(frameCopy,(c[0],c[1]),(c[2],c[3]),(255,0,0),thickness=-1)
    
    # list of boxes that have corresponding boxes in previous frame
    i_list=matchboxes(coordlist, prev_coordlist, width)
    
    # get list of boxes that are new in the frame
    new_list=newbox(coordlist,i_list)
    
    # get list of boxes that have disappeared
    disp_list=dispbox(prev_coordlist,i_list)
    
    # adjust number of people and draw rectangles
    for new_coords in new_list:
        if checkbot_box(new_coords,height)==1:
            num_people-=1
            cv2.rectangle(frameCopy,(new_coords[0],new_coords[1]),(new_coords[2],new_coords[3]),(0,0,255),thickness=-1)
    
    for disp_coords in disp_list:
        if checkbot_box(disp_coords,height)==1:
            num_people+=1
            cv2.rectangle(frameCopy,(disp_coords[0],disp_coords[1]),(disp_coords[2],disp_coords[3]),(0,255,0),thickness=-1)
    
    # add the rectangles to the frame
    frame=cv2.addWeighted(frameCopy,0.3,frame,0.7,1.0)

    return frame, num_people



# import RPi.GPIO as GPIO
# GPIO.setmode(GPIO.BCM)
# pin_num = 21
# GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)

def GPIO_LIGHT(numPeople, frame):
    # if numPeople > 0: GPIO.output(pin_num, GPIO.HIGH)
    # else: GPIO.output(pin_num, GPIO.LOW)

    if numPeople > 0: cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(255,255,255), thickness=cv2.FILLED)
    else: cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=30, color=(0,0,0), thickness=cv2.FILLED)      
    


while(1):
    frameno+=1
    _, frame = cap.read()
    
    # create frames for color filling in
    rect_frame=frame.copy()


    results = model(frame)
    if frameno==1:
        prev_results=results
    

    frame, num_people = COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)

    # send rasp GPIO command  
    GPIO_LIGHT(num_people, frame)


    fpsEnd = time.time()
    timeDiff = fpsEnd - fpsStart
    fps = 1/timeDiff
    fpsStart = fpsEnd

    fpsText  = "FPS: {:2.2f}".format(fps)
    cv2.putText(frame, fpsText, (30, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)    

    num_peopletxt="Number of people: "+str(num_people)
    cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_SIMPLEX, round(width/1000), (0, 0, 255), round(width/1000), cv2.LINE_AA)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    
    prev_results=results
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        # GPIO.output(pin_num, GPIO.LOW)
        # GPIO.cleanup()
        break
    if k == 114 or k == 82:
        num_people = 0
```

**Code to test on Raspberry Pi**

```python
import torch
import cv2
import random
from PIL import Image
import numpy as np
import math
import time

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
model.classes=[0]


cap = cv2.VideoCapture(0)
width = int(cap.get(3)); height = int(cap.get(4)); 

frameno=0
num_people=0
fpsStart = 0
fps = 0



# returns coordinates of box as list
def box_coords(box):
    xmin=int(box[0])
    ymin=int(box[1])
    xmax=int(box[2])
    ymax=int(box[3])
    return [xmin, ymin, xmax, ymax]

# checks if box touches the bottom of frame
def checkbot_box(coords,height):
    ymax=coords[3]
    if ymax>height-(height/54):
        return 1
    else:
        return 0

# returns center coordinates of box
def box_cent(coords):
    cent_x=int((coords[0]+coords[2])/2)
    cent_y=int((coords[1]+coords[3])/2)
    return [cent_x,cent_y]

# gets intersecting area of two boxes
def inters_area(coord1,coord2):
    xmin1=coord1[0]
    ymin1=coord1[1]
    xmax1=coord1[2]
    ymax1=coord1[3]
    xmin2=coord2[0]
    ymin2=coord2[1]
    xmax2=coord2[2]
    ymax2=coord2[3]
    dx=min(xmax1,xmax2)-max(xmin1,xmin2)
    dy=min(ymax1,ymax2)-max(ymin1,ymin2)
    if (dx>0) and (dy>0):
        return dx*dy
    else:
        return 0

# returns list of coordinates of boxes in current frame that are new (no corresponding box in previous frame)
def newbox(coordlist,i_list):
    new_list=[]
    for k in coordlist:
        if k not in [i[0] for i in i_list]:
            new_list+=[k]
    return new_list

# returns list of coordinates of boxes in previous frame that have disappeared (no corresponding box in current frame)
def dispbox(prev_coordlist,i_list):
    disp_list=[]
    for k in prev_coordlist:
        if k not in [i[1] for i in i_list]:
            disp_list+=[k]
    return disp_list

# finds which box in previous slide is the one in current frame (highest intersecting area)
def matchboxes(coordlist,prev_coordlist,width):
    i_list=[]
    for coord in coordlist:
        area=0
        add_ilist=[]
        for prev_coord in prev_coordlist:
            if inters_area(coord,prev_coord)>area and (math.dist(box_cent(coord),box_cent(prev_coord))<(4*width/20)):
                area=inters_area(coord,prev_coord)
                add_ilist=[[coord, prev_coord]]
            if coord not in [i[0] for i in i_list] and prev_coord not in [j[1] for j in i_list]:
                i_list+=add_ilist
    return i_list


# COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)
def COUNT_PEOPLE_FRAMEOUT(dataPre, dataCur, frame, frameCopy, num_people):
    # create lists of all box coordinates in previous and current frame
    prev_coordlist=[]
    for j in range(len(dataPre.xyxy[0])):
        prev_coords=box_coords(dataPre.xyxy[0][j])
        prev_coordlist+=[prev_coords]
    coordlist=[]
    for k in range(len(dataCur.xyxy[0])):
        coords=box_coords(dataCur.xyxy[0][k])
        coordlist+=[coords]
    
    for c in coordlist:
        cv2.rectangle(frameCopy,(c[0],c[1]),(c[2],c[3]),(255,0,0),thickness=-1)
    
    # list of boxes that have corresponding boxes in previous frame
    i_list=matchboxes(coordlist, prev_coordlist, width)
    
    # get list of boxes that are new in the frame
    new_list=newbox(coordlist,i_list)
    
    # get list of boxes that have disappeared
    disp_list=dispbox(prev_coordlist,i_list)
    
    # adjust number of people and draw rectangles
    for new_coords in new_list:
        if checkbot_box(new_coords,height)==1:
            num_people-=1
            cv2.rectangle(frameCopy,(new_coords[0],new_coords[1]),(new_coords[2],new_coords[3]),(0,0,255),thickness=-1)
    
    for disp_coords in disp_list:
        if checkbot_box(disp_coords,height)==1:
            num_people+=1
            cv2.rectangle(frameCopy,(disp_coords[0],disp_coords[1]),(disp_coords[2],disp_coords[3]),(0,255,0),thickness=-1)
    
    # add the rectangles to the frame
    frame=cv2.addWeighted(frameCopy,0.3,frame,0.7,1.0)

    return frame, num_people





import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
pin_num = 21
GPIO.setup(pin_num, GPIO.OUT, initial=GPIO.LOW)

def GPIO_LIGHT(numPeople, frame):
    if numPeople > 0: GPIO.output(pin_num, GPIO.HIGH)
    else: GPIO.output(pin_num, GPIO.LOW)

    cv2.circle(frame, (int(width*0.9), int(height*0.9)), radius=31, color=(0,0,0), thickness=cv2.FILLED)      
    if numPeople > 0: 
        cv2.putText(frame, 'ON' ,(int(width*0.865), int(height*0.92)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)    
    




    
resultFINAL = cv2.VideoWriter('demovideo.avi', cv2.VideoWriter_fourcc(*'XVID'), cap.get(cv2.CAP_PROP_FPS), (width, height)) # 3 is FPS / cap.get(cv.CAP_PROP_FPS)

while(1):
    frameno+=1
    _, frame = cap.read()
    
    # create frames for color filling in
    rect_frame=frame.copy()


    results = model(frame)
    if frameno==1:
        prev_results=results
    


    frame, num_people = COUNT_PEOPLE_FRAMEOUT(prev_results, results, frame, rect_frame, num_people)

    # send rasp GPIO command  
    GPIO_LIGHT(num_people, frame)


    fpsEnd = time.time()
    timeDiff = fpsEnd - fpsStart
    fps = 1/timeDiff
    fpsStart = fpsEnd

    fpsText  = "FPS: {:2.2f}".format(fps)
    cv2.putText(frame, fpsText, (int(width/40), int(height/15)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)    

    num_peopletxt="Number of people entered: "+str(num_people)
    if num_people>0:
        cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, num_peopletxt, (int(width/40), height-int(width/40)), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.imshow("result", frame)
    

    resultFINAL.write(frame)


    prev_results=results
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        GPIO.output(pin_num, GPIO.LOW)
        GPIO.cleanup()
        break
    if k == 114 or k == 82:
        num_people = 0


cap.release()
resultFINAL.release()

cv2.destroyAllWindows()
```

## Reference

#### Code Reference

* https://ykkim.gitbook.io/dlip/
* https://github.com/ultralytics/yolov5
* some class materials from ECE30003
