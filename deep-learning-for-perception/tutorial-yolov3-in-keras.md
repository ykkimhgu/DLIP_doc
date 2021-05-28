# Tutorial:  Yolov3 in Keras

### Reference

Github:  [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)



### Create Virtual Environment \(Conda\)

Lets create a virtual environment for YOLOv3.

The requirements are

* python=3.7.10
* cudatoolkit=10.0
* cudnn-7.6.5-cuda10.0\_0
* tensorflow-gpu=1.15.0
* keras=2.3.1
* pillow=8.2.0
* matplotlib=3.3.4
* opencv=3.4.2

 If you have problems when installing opencv packages, use the following commands `pip install opencv-python`

```text
conda create -n tf115 python=3.7
conda activate tf115
```

Install the following:

```text
conda install cudatoolkit=10.0
conda install cudnn
conda install tensorflow-gpu=1.15.0
conda install keras=2.3
conda install pillow
conda install matplotlib
conda install opencv
```



### Clone Git

After the installation, activate the virtual environment. We will clone the reference repository to download Yolov3 codes.

#### Method 1:  From conda prompt \(in virtual env\)

git [https://github.com/qqwweee/keras-yolo3.git](https://github.com/qqwweee/keras-yolo3.git)

#### Method 2:

Download as zip file from the github and unzip.

![](../.gitbook/assets/image%20%28318%29.png)

### Download the trained weight file

After download, place the weight model in the same directory of Yolov3.

*  YOLOv3 weights [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

> You can also download from Prompt as

`wget` [`https://pjreddie.com/media/files/yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)\`\`

*  YOLOv3-tiny weights

[https://pjreddie.com/media/files/yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)

### Run V.S Code

 `>> code .`

> You can also run the below codes in Conda Promt

### Convert Darknet YOLOv3 to Keras model

Using the virtual env, in the termnial type:

```text
>> python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```



### Run Yolov3 Detection

If the test video is in the same directory and the name is 'test\_Video.avi'

```bash
python yolo_video.py --model .\model_data\yolo.h5 --input .\test_Video.avi
```

`python yolo_video.py --model .\model_data\yolo/h5 --input .\testVideo.mp4`

\`\`

#### Usage

Use --help to see usage of yolo\_video.py:

```text
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```



