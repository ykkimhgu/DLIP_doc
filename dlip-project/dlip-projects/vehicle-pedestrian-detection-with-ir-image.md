# Vehicle, Pedestrian Detection with IR Image

**Date:** 2021-6-21

**Author**:  김도연, 이예

**Github:**&#x20;

**Demo Video:**

## Introduction

This tutorial explains the FLIR image object detection using yolov5. If you want to know how to install yolov5 in your desktop, referring to this site [https://ropiens.tistory.com/44](https://ropiens.tistory.com/44).

This report consists of five parts.

* Why do I use FLIR camera
* Train FLIR dataset and obtain the training 'weight'
* The simple usage of FLIP camera (FLIR A60)
* Pre-processing codes
* Discussion

Wanting to know only the way to train dataset, you can only refer to step 2, (Train FLIR dataset and obtain the training 'weight')

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FxaCRmYHsvKcWQvzMz7gZ%2Ffile.png?alt=media)

## 1.Why do I use FLIR camera

* At night, it is hard to object detection because it is difficult to distinguish objects from common cameras.
* But FLIR camera not affected by light
* These camera can see through smoke, fog, haze, and other atmospheric obscurants better than a visible light camera can.

### It is useful at night

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FTsDjVqyA3LJkEE4nnIWG%2Ffile.png?alt=media)

### 2. It is not affected by visible light.

![light\_sum.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FkklBWZuZ1mdvfXjnVxCf%2Ffile.png?alt=media)

## 2.Train FLIR dataset and obtain the training 'weight'

### **Download yolov5 from Github!!!**

```python
%cd /content
!git clone https://github.com/ultralytics/yolov5.git

%cd /content/yolov5/
!pip install -r requirements.txt
```

### **Download dataset from Google drive**

Before you get your dataset from Google drive, you should upload your data in your own Google drive.

For example, in this tutorial, I made data folder named 'juho' in my google drive. In 'juho', there are 'images' folder which has training image set and validateion image set and 'lables' folder which has lables of training images and labels of validation images.

Figure below shows the subfolers of 'juho' folder!

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FLcmCoaEcLfGC0jEMugsQ%2Ffile.png?alt=media)

```python
from google.colab import drive 
drive.mount('/content/yolov5/drive')
```

```
Mounted at /content/yolov5/drive
```

If you run this and login your ID, you will get the authorizing code

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FeqFs4DbjZaY6KPF7Y1hG%2Ffile.png?alt=media)

Then you can get the "drive" folder and you can access your data in Google drive

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F3F87wIZD78yOjKWfwT6d%2Ffile.png?alt=media)

### **Check yaml file**

'yaml' file includes the imformation of your dataset.

* 'train' in 'yaml' indicates the directory of your training images!
* 'val'  in 'yaml' indicates the directory of your validating images!
* 'nc' means the number of class that your data file detecting
* 'names' means the sort of classes your data file detecting

```python
 %cat /content/yolov5/drive/MyDrive/juho/data.yaml
```

```
train: ../train/images
val: ../valid/images

nc: 4
names: ['1', '18', '2', '3']
```

### Making image lists (Test and Validation)

This stage is making image lists for your image data!

* 'train\_img' is a training image list
* 'val\_image' is a validating image list

In this tutorial, the image files used are '.jpg' format, you can change it to your own file format!

### And make sure your data is uploaded successfully!

```python
%cd
from glob import glob

train_img = glob('/content/yolov5/drive/MyDrive/juho/images/train_images/*.jpg')
val_img   =glob('/content/yolov5/drive/MyDrive/juho/images/val_images/*.jpg')
train_img = train_img[0:1500]
val_img   = val_img[0:300]

print("Train images:",len(train_img))
print("Validating images:",len(val_img))

import matplotlib.pyplot as plt
import random
import matplotlib.image as Image

testnum    = random.randrange(0,1501)
test_img   = train_img[testnum]
img = Image.imread(test_img)
imgplot = plt.imshow(img)
plt.show()
```

```
/root
Train images: 1500
Validating images: 300
```

### Making path text files (Test and Valition)

You should modify directory of your 'train\_img' and 'val\_img' in 'yaml' file (in Check yaml file session). Because directories of data files changed in Google drive. This code make the directory (path of your files) as a .txt file.

```python
with open('/content/yolov5/data/train.txt','w') as f:
  f.write('\n'.join(train_img)+'\n')
with open('/content/yolov5/data/val.txt','w') as f:
  f.write('\n'.join(val_img)+'\n')
```

### Check there are new txt files ('train.txt' and 'val.txt') in '/content/yolov5/data/'.

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FHfGpr3t3P4ZdwHsU1WTi%2Ffile.png?alt=media)

### Modifying the data.yaml file

Using the 'train.txt' and 'val.txt'(in Making path text files session), you can change your 'train' and 'val' directory in 'yaml' file.

```python
import yaml

with open('/content/yolov5/drive/MyDrive/juho/data.yaml','r') as f:
  data = yaml.load(f)

print(data)

data['train'] ='/content/yolov5/data/train.txt'
data['val']  = '/content/yolov5/data/val.txt'

with open('/content/yolov5/data/data.yaml','w') as f:
 yaml.dump(data,f) 

print(data)
```

```
{'train': '../train/images', 'val': '../valid/images', 'nc': 4, 'names': ['1', '18', '2', '3']}
{'train': '/content/yolov5/data/train.txt', 'val': '/content/yolov5/data/val.txt', 'nc': 4, 'names': ['1', '18', '2', '3']}


/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:4: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  after removing the cwd from sys.path.
```

### **Train your data and obtain the weight file!!!**

This stage is 'training stage'. If you run this code, you can get a 'weight' file for your own dataset. There are some parameters you should know.

* '--img'     : the image size of your data
* '--batch'   : set of data calculated per process.&#x20;
* '--epochs'  : total number of data training &#x20;
* '--data'    : the directory of your 'yaml' data file
* '--cfg'     : the structure of the weight model(in this tutorial 'yolo5x.yaml' used)&#x20;
* '--weights' : training weight (in this tutorial 'yolo5x' used)&#x20;
* '--name'    : the name of your own weight file, the result of this training

```python
%cd /content/yolov5/

!python train.py --img 416 --batch 8 --epochs 50 --data /content/yolov5/data/data.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name FIRL_yolov5x
```

### **Check your training result!**

```python
import numpy as np 
from  PIL import Image

path = "/content/yolov5/runs/train/FIRL_yolov5x/results.png"
fig = plt.figure()

result_img   =Image.open(path)  
result_img   = result_img.resize((2048*2,2048*2))
imgplot = plt.imshow(result_img)
plt.show()
```

### Move your weight file to your yolov5 in desktop

After finishing your custom training dataset,you should move the 'weight' file to your desktop The result weight file is located at '/content/yolov5/runs/train/FIRL\_yolov5x/weights/ (FIRL\_yolov5x is the weight file name used in this tutorial) ![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FzfWLEFM7b94Gnty6B1U7%2Ffile.png?alt=media)

You can move 'best.pt' file to your desktop YOLOv5 folder.

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F2QBQniHiF7DSjvq7aPts%2Ffile.png?alt=media)

Finally, you can run your code with your own weight file!!!

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FZv0FOiNoxAtNRIlbYkVB%2Ffile.png?alt=media)

## 3.The simple usage of FLIP camera (FLIR A65)

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FOi1Klpkvcyi6OcgEofyg%2Ffile.png?alt=media)

image from [https://www.flirkorea.com/products/a65/](https://www.flirkorea.com/products/a65/)

### Download FLIR software!!

you can download FLIR software from [https://go.pleora.com/Download-eBUS-Player](https://go.pleora.com/Download-eBUS-Player)

download FLIR GEV Dem 1.10.0 version ![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FOJTiyAoEd7W6UinlxO9W%2Ffile.png?alt=media) After download this software

Install the ebus\_runtime\_32-bit.4.1.6.3809.exe and run the PvSimpleUISample.exe

After, click the Select/Connect button, then connect your FLIR camera

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2Fl02exwzsV13fx6Mt8KCe%2Ffile.png?alt=media)

## 4.Pre-processing code

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FCX3Q6n3ZBUUVuhAog8tT%2Ffile.png?alt=media)

### Add pre-processing code in yolov5

Anaconda Prompt allows dircet access to the yolov5 file to modify the code. It can modify overall extent of yolov5 before object detection, such as what object to find. We did the following setting: 1. Set Roi to detect only the objects of lane where this vehicle is located 2. Approximate distance measurement - Display warnining message about 2m when a person or car in front of the car

### 1. Set Roi

![roi.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FlgefW8AiQrXIHOqLS6VL%2Ffile.png?alt=media)

![set roi.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FE4OnadVJLEiqvF2E8fNE%2Ffile.png?alt=media)

Set roi region like this, in order to get the information needed only from the front of the vehicle when driving.

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FEeqSmQdBEMuViS5jPJ1X%2Ffile.png?alt=media)

If the object is not in range, do not detect it. More clear detection is possible by blocking unnecessary information (such as vehicles in other lanes).

### 2. Display warning message

![warning text.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F5z9rWpUSEnUhgM7Xnh8z%2Ffile.png?alt=media)

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2Fb87FOdVoGjsZAapMvtp2%2Ffile.png?alt=media)

If a person or other vehicle is too close to the front of the vehicle, warning message is displayed on the screen. The distance was set at approximately 2 m.

In detect.py code, the xyxy variable is the coordinate value of the upper left and lower right points of the bounding box being detected. This allows the warning message to be displayed when the y-value of the lower right-hand spot is 30 percent from the bottom of the screen, which is thought to be 2 meters from the camera.

## 5.Dicussion

In this step, the overall assesment wil be conducted.

### 1. Lack of detection capability of our custom trained weight file

if you look at the above figure, it can be seen that there are three objects(1 person ,2 cars),but actually there are only two objects(1 person, 1 car). It means that it detect the wrong object. There are mainly two reason for this error.

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FGCeZaUd6wXJkLhRzHfnQ%2Ffile.png?alt=media)

* Environmental effect

The FLIR measures the infra wave from the object. Therefore, it is affected the environment of the place where images are taken. the training images are taken in California , but the test images are in Pohang. For this reason, this error occurs.

*   Filter effect

    In FLIR camera setting, there is a filter effect. The result of the picture changes depending on which filters user uses. The difference filter effect between training data and testing data make this error.&#x20;

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2Fy3DQYQc1bWWUqcBp92CV%2Ffile.png?alt=media)

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F75iskT4Doq8e7OYDLLmY%2Ffile.png?alt=media)

The picture above and the picture below are pictures with different filters. Because FLIR images are represented by binaries, these filter differences make a very big difference.

### 2. Unable to measure distance accurately

The distance measured in this project (2 meters apart from the camera) was about 30 percent of the screen. But this is a very rough value, and it actually needs to be more precise.

* When object detection is performed, the bounding box is not exactly fitted to the object.&#x20;
* Only measured values can always be used in fixed positions because the camera can vary at all times depending on the height, angle, etc.

Therefore, we suggest utilizing a combination of sensors other than FLIR cameras (e.g., lidar sensors that read reflected values using laser pulses) to measure the distance precisely.
