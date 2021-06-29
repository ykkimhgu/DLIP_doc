# Parking Space Management

**Date:** 2021-6-21

**Author**:  강지훈, 도경민 

**Github:** [https://github.com/chloerudals/DLIP\_LAB4](https://github.com/chloerudals/DLIP_LAB4)

**Demo Video:**

{% embed url="https://www.youtube.com/watch?v=og6CcAu\_-JY" %}

\*\*\*\*

## Introduction

Since the use of private vehicle has increased due to COVID-19, finding parking spaces has been difficult even in our university. Thus, we decided to show the empty parking spaces on the screen to make parking management easier.

![](../../.gitbook/assets/image%20%28323%29.png)

![parking lot](https://github.com/chloerudals/DLIP_LAB4/blob/main/Img.jpg)  
 To watch a short explanatory video. [Click Here](https://youtu.be/og6CcAu_-JY)

> Algorithm: 1. Parking lines are detected using HoughlinesP and cars are detected using Yolov5s 2. We improved the detection of parking lines, which had previously been covered by parked cars, by elongating the lines 3. We divided the rectangle the same number as the parking lot. 4. Adjusted distorted regions due to perspectives. 5. By comparing the center of the parking space and the center of the detected box, parking ability is decided. 6. Since cars park in the upper part of the parking space, y axis of the detected box's center is corrected about 10 pixels 7. If a car comes in the camera frame, the car is considered as parking so entering car is printed.

## Requirements

1. Hardware

   > * NVDIA graphic cards

2. Software

   > * CUDA
   > * cuDNN
   > * Anaconda
   > * YOLO V5

Or You can follow the instructions from the yolov5 GitHub repository. [\(requirements\)](https://github.com/ultralytics/yolov5/blob/master/requirements.txt)

### Follow the steps for setting YOLO V5 in the Anaconda.

> Before starting, check your gpu to match the version.
>
> ```text
>   # create a conda env name=yolov5 (you can change your env name)
>   conda create -n yolov5 python=3.8
>   conda activate yolov5
>   
>   # Installation process
>   conda install cudatoolkit=10.2
>   conda install cudnn
>   conda install tensorflow-gpu=2.3.0
>   conda install keras
>   conda install pillow
>   conda install matplotlib
>   conda install opencv
>   
>   # clone yolov5
>   git clone https://github.com/ultralytics/yolov5.git
>   
>   # update
>   conda update -yn base -c defaults conda
>   
>   # install Lib for YOLOv5
>   conda install -c anaconda cython numpy pillow scipy seaborn pandas requests
>   conda install -c conda-forge matplotlib pyyaml tensorboard tqdm opencv 
> ```
>
> ```text
>   # install pytorch
>   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
>  
>   # Extra
>   conda install -c conda-forge onnx
> ```

## Essential codes to understand the program.

#### Finding the parking lines.

> * First, you need a parking lot's picture to detect the parking lines. \(an empty parking lot image would be perfect.\)
>
>   * Image processing is conducted.
>
>     > Since the lines are mostly white and yellow, select only yellow and white colors from the image.
>
>   > def select\_rgb\_white\_yellow\(image\):
>
>   ![white-yellow](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/white_yellow_image.jpg)
>
>   ![white-yellow](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/white_yellow_image.jpg)
>
>   > Convert the image to gray scale.
>   >
>   > ```text
>   >   def convert_gray_scale(image):
>   > ```
>   >
>   > ![gray](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/gray_image.jpg)
>
> > Detect the edges with _**Canny**_.
> >
> > ```text
> >   def detect_edges(image, low_threshold=500, high_threshold=1000):
> > ```
> >
> > ![detect edges](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/edge_image.jpg)
> >
> > * Crop the image using _**roi**_.
> >
> >   def filter\_region\(image, vertices\): def select\_region\(image\):
> >
> >   ![roi](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/roi_image.jpg)
> >
> > * Using HoughlinesP, detect the vertical parking lines.
> >
> >   def hough\_lines\(image\): def draw\_lines\(image, lines, color=\[255, 0, 0\], thickness=2, make\_copy=True\):
> >
> >   ![line](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/line_image.jpg)
> >
> > * Draw a rectangle.
> >
> >   def identify\_blocks\(image, lines, make\_copy=True\):
> >
> >   ![rect](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/rect_image.jpg)
> >
> > * Adjust the rectangle.
> > * Merge the rectangle with the adjusted verticle lines to delineate the parking lines.
> > * Count the total parking spaces.
> >
> >   def draw\_parking\(image, rects, make\_copy = True, color=\[255, 0, 0\], thickness=1, save = True\):
> >
> >   ![delineated\_image](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/delineated_image.jpg)
> >
> > * Assign a number to the parking spaces.
> >
> >   def assign\_spots\_map\(image, spot\_dict=final\_spot\_dict, make\_copy = True, color=\[255, 0, 0\], thickness=2\):
> >
> >   ![marked\_spot\_images](https://github.com/chloerudals/DLIP_LAB4/blob/main/images/marked_spot_images.jpg)

To see a detailed explanation. [Click Here](https://github.com/chloerudals/DLIP_LAB4/blob/main/identify_parking_spots.ipynb)

#### Detect cars.

> * Car detection is done by YOLO V5s with COCO datasets.

#### Distinguish whether the parking space is empty or not.

> * Firstly, find the centers of the parking space and the car.
>   * If the distance between the centers are less than 40, the parking space is determined as occupied.

```text
for *xyxy, conf, cls in reversed(det):
    bx1, by1, bx2, by2 = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
    bcx, bcy = abs(bx2 + bx1) / 2, abs(by2 + by1) / 2 + 10
    cv2.circle(im0, (int(bcx), int(bcy)), 5, (255, 255, 255), 2)
    parking_distance = math.sqrt((bcx - pcx)**2 + (bcy - pcy)**2)

    if parking_distance < 40:
        cv2.polylines(im0, [poly_points], 1, [0, 0, 255], 2)
        cv2.line(im0, (int(bcx), int(bcy)), (int(pcx), int(pcy)), (255, 255, 255), 2)
        cv2.putText(im0, "%d" %spot_cnt, text_coordinate, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        Parked_Space_Array.append(spot_cnt)
        break
```

#### Editing Parser

> * Since we are using 'YOLO V5s model', we set the default for weights as 'yolov5s'. 
>   * Image size is set = 608
>   * conf-thres value = 0.3
>   * iou-thres value = 0.4
>   * view-img's action is set as 'store\_false' to view the image.
>   * save-txt's action is set as 'store\_false' to save the result.
>   * classes's default is added as 2 to only view car class.
>   * The bounding box's line-thickness is edited to 2.
>
>        parser.add\_argument\('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path\(s\)'\)
>
>        parser.add\_argument\('--img-size', type=int, default=608, help='inference size \(pixels\)'\)
>
>        parser.add\_argument\('--conf-thres', type=float, default=0.3, help='object confidence threshold'\) 
>
>        parser.add\_argument\('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS'\)
>
>        parser.add\_argument\('--view-img', action='store\_false', help='display results'\)
>
>        parser.add\_argument\('--save-txt', action='store\_false', help='save results to \*.txt'\)
>
>        parser.add\_argument\('--classes', nargs='+', type=int, default=2, help='filter by class: --class 0, or --class 0 2 3'\)
>
>        parser.add\_argument\('--line-thickness', default=2, type=int, help='bounding box thickness \(pixels\)'\)

[Demo Video](https://drive.google.com/file/d/1LPtyEVEorxBqGS-NXqe6Ns1JMTjdhgKB/view?usp=sharing)

## How to run the program

1. Download [video](https://drive.google.com/file/d/170Ccn_BTxPyWlN8Trfk9KXK6ykQmQNAW/view?usp=sharing) to your _**yolov5**_ repository.
2. Download [image](https://github.com/chloerudals/DLIP_LAB4/blob/main/Img.jpg) to your _**yolov5\data\images**_ directory.
3. Overwrite [detect.py](https://github.com/chloerudals/DLIP_LAB4/blob/main/detect.py) to the _**yolov5**_ repository.
4. Overwrite [datasets.py](https://github.com/chloerudals/DLIP_LAB4/blob/main/datasets.py) to your _**yolov5\utils**_ directory.

## Future work

* Detect the parking lines without conducting parking space image and adjusting the lines.
* Specify the vehicles whether they are entering, leaving or staying for a minute.
* Give an alarm to the securitary, if there is a double-parked car.

## Reference

[dudesparsh: Parking detector](https://github.com/dudesparsh/Parking_detector)

