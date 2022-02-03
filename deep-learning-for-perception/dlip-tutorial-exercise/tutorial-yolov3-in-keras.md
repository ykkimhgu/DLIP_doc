# Tutorial:  Yolov3 in Keras

## Reference

Github:  [https://github.com/qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3)



## Setup

### Create Virtual Environment (Conda)

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

&#x20;If you have problems when installing opencv packages, use the following commands `pip install opencv-python`

```bash
conda create -n tf115 python=3.7
conda activate tf115
```

Install the following:

```bash
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

#### Method 1:  From conda prompt (in virtual env)

git [https://github.com/qqwweee/keras-yolo3.git](https://github.com/qqwweee/keras-yolo3.git)

#### Method 2:

Download  zip file from the github and unzip.

![](<../../.gitbook/assets/image (318).png>)

### Download the trained weight file

After the download, place the weight model file in the same directory of Yolov3.

* &#x20;YOLOv3 weights\
  [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)

> You can also download it from the conda Prompt as

`wget` [`https://pjreddie.com/media/files/yolov3.weights`](https://pjreddie.com/media/files/yolov3.weights)``

* &#x20;YOLOv3-tiny weights

[https://pjreddie.com/media/files/yolov3-tiny.weights](https://pjreddie.com/media/files/yolov3-tiny.weights)

### Open V.S Code

&#x20;`>> code .`

> You can also run the below codes in the Conda Promt

In VS code, select the virtual environment:  F1--> Python Interpreter --> Select Environ.

### Convert Darknet YOLOv3 to Keras model

In the terminal of VS code or  in Conda Prompt, type:

```bash
>> python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```



## Run Yolov3 Detection

Copy the test video file in the same directory (Yolov3 directory)

If the video file name is 'test\_Video.avi'

```bash
>> python yolo_video.py --model .\model_data\yolo.h5 --input .\test_Video.avi
```

![](<../../.gitbook/assets/image (319).png>)

### Run Yolov3-Tiny Detection

After downloading yolov3-tiny.weights,  Convert it to Keras model and save it as 'yolo-tiny.h5'

```bash
 >> python convert.py yolov3-tiny.cfg yolov3-tiny.weights model_data/yolo-tiny.h5
```

Run Yolo-tiny with the Test video

```bash
>> python yolo_video.py --model .\model_data\yolo-tiny.h5 --input .\test_Video.avi
```

### Usage

Use --help to see usage of yolo\_video.py:

```
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



## How to train a dataset

### Prepare the dataset

For this tutorial, we will use KITTI dataset

* Image file: [Download Kitti Dataset](https://s3.eu-central-1.amazonaws.com/avg-kitti/data\_object\_image\_2.zip)
* Label file:  [Download Kitti Dataset Label](https://s3.eu-central-1.amazonaws.com/avg-kitti/data\_object\_label\_2.zip)
  * Object Detection annotation Convert to Yolo Darknet Format: [Click here](https://github.com/ssaru/convert2Yolo)
* Class file:&#x20;
  * Copy the 'kitti\_classes.txt'  in  the folder of  \`\model\_data\` folder

### Modify train.py&#x20;

Open 'train.py' file  in VS Code\


Go to  LIne 16 : def _main():._  Change the '_'annotation' and 'classes-path'_  to your setting.

```python
def _main(): 
annotation_path = 'train.txt' log_dir = 'logs/000/'
#classes_path = 'model_data/voc_classes.txt'
classes_path = 'model_data/kitti_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
```

Go to LIne 32:  Change the name of the pre-trained weight file.

* We will use COCO trained weight file as we used above(yolo.h5).  Create  a copy and name it as`yolo_weights.h5`

```python
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_tiny_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/yolo_weights.h5') # make sure you know what you freeze

```

### Run Train

Start training by running the following in the terminal

```bash
>>python train.py
```



### Evaluate

Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo\_video.py Remember to modify the class path or anchor path.



## TroubleShooting

### Problem 1

Error message of

`_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])`&#x20;

#### Solution

Modify  `model.py` (line 394)

`_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])`&#x20;

should be changed to&#x20;

`_, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])`

