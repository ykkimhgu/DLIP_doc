# Tutorial: YOLO



## Tutorial: Installation and Inference Examples

### Introduction

Tutorial: YOLO26 in PyTorch

YOLO26 is a cutting-edge model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility.

![img](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fraw.githubusercontent.com%2Fultralytics%2Fassets%2Fmain%2Fyolov8%2Fyolo-comparison-plots.png\&width=768\&dpr=3\&quality=100\&sign=a45b1c87\&sv=2)

* Documentation: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
* GitHub: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

***

### Installation

Official Installation Guide:  [https://docs.ultralytics.com/quickstart/#install-ultralytics](https://docs.ultralytics.com/quickstart/#install-ultralytics)

#### Requirement

Install necessary packages such as Python, Numpy, PyTorch, CUDA and more

For installations of requirements, read for more detail instructions

* Python >=3.10
* PyTorch, TorchVision, TorchAudio
* CUDA version compatible with your NVIDIA driver
* ultralytics latest stable release
* opencv-python
* matplotlib
* torchsummary, onnx, pytubefix
* NumPy 1.26 only if needed for an OpenCV DLL issue

#### Install YOLO26 via pip package

First, create a new environment for YOLO26 in Anaconda Prompt.

Check the PyTorch installation command and CUDA Toolkit / NVIDIA driver table:

* PyTorch installation: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
* CUDA Toolkit and driver version table: [https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-driver)

```batch
conda create -n yolo26 python=3.10
conda activate yolo26

conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

python -m pip install --upgrade pip

pip install -U ultralytics
pip install opencv-python matplotlib torchsummary onnx pytubefix
```

The example codes save result images and display them with `cv2.imshow()`, so make sure the `yolo26` environment is activated and `opencv-python` is installed.



**If there is an OpenCV DLL issue:**

* Follow the NumPy 1.26 setup instructions below to configure your environment.

```bat
conda install -c anaconda numpy=1.26
pip install --force-reinstall opencv-python
```

### Check YOLO Installation

After the installation, you can check the saved source code and libs of YOLO26 with:

```bat
python -c "import ultralytics, pathlib; print(pathlib.Path(ultralytics.__file__).parent)"
```

The result will be similar to:

```
%USERPROFILE%\anaconda3\envs\yolo26\Lib\site-packages\ultralytics
```

### **Demo: Detection and Segmentation Inference**

#### **Detection Demo**

Now, let's run simple prediction examples to check the YOLO installation.

In Anaconda Prompt, activate `yolo26` environment.

Then, move directory to the working directory. Here, the result images will be saved.

* Example: `C:\Users\ykkim\source\repos\DLIP\YOLOv26\`

```bat
conda activate yolo26
cd C:\Users\ykkim\source\repos\DLIP\YOLOv26
```



In the Anaconda prompt, type the following command to predict a simple image.

```bat
yolo predict model=yolo26n.pt source='https://ultralytics.com/images/bus.jpg'
```

The result will be saved in the project folder `\runs\detect\predict\`

> Example: C:\Users\ykkim\source\repos\DLIP\YOLOv26\runs\detect\predict\\

<img src="https://github.com/user-attachments/assets/0582fad0-0f93-4466-b630-f47c09e5396e" alt="YOLO26 detection result example" width="560">

#### **Segmentation Demo**

Predict a YouTube video using a pretrained segmentation model at image size 320:

```bat
yolo predict model=yolo26n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
```

This YouTube example runs the full video, so it may take several minutes.

The result will be saved in the project folder `\runs\segment\predict\`

<img src="https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2F3883264845-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252F-MR8tEAjhiC8uN1kHR2J%252Fuploads%252FRZVFMqeDlRDPLgz4nzxB%252Fimage.png%3Falt%3Dmedia%26token%3D68f882ef-bf97-4e2b-a632-8281ae9800fd&#x26;width=768&#x26;dpr=3&#x26;quality=100&#x26;sign=da72ae76&#x26;sv=2" alt="" width="375">

***

### Example 1: Detection Inference (one image)

#### Preparation

In the project folder, create a new python code file

* Project Folder: \source\repos\DLIP\YOLOv26\\
* Activate `yolo26` environment in Anaconda Prompt

A list of useful commands for YOLO26

```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolo26n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolo26n.pt')

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data='coco8.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')
```

Download a sample image and save it as `bus.jpg` in the project folder. For the multiple-image example, copy it as `bus2.jpg`.

```bat
curl -L -o bus.jpg https://ultralytics.com/images/bus.jpg
copy bus.jpg bus2.jpg
```

You can download the COCO pretrained models such as YOLO26n and more.

#### **Inference one image**

Create a new python source file in the project folder

* `Yolo26-Inference-Ex1.py`

```python
# YOLO26 Tutorial : Prediction Ex1
# Load Pretrained Model and Display the Annoted Results

from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolo26n.pt')

# Inference Source - a single source
src = cv2.imread("bus.jpg")
if src is None:
    raise FileNotFoundError("bus.jpg")

# Perform object detection on an image using the model
result = model.predict(source=src, save=True, save_txt=True)  # save predictions as labels

# View result
for r in result:

    # print the Boxes object containing the detection bounding boxes
    print(r.boxes)

    # Plot results image
    print("result.plot()")
    dst = r.plot()      # return BGR-order numpy array
    cv2.imwrite("result_plot.jpg", dst)
    cv2.imshow("result plot", dst)

    # Plot the original image (NParray)
    print("result.orig_img")
    cv2.imwrite("result_orig.jpg", r.orig_img)
    cv2.imshow("result orig", r.orig_img)

    # Save results to disk
    r.save(filename='result.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Example 2: Detection Inference (multiple images)

Create a new python source file in the project folder

* `Yolo26-Inference-Ex2.py`

For multiple input source images, you can copy `bus.jpg` as `bus2.jpg`.

```python
# YOLO26 Tutorial : Prediction Ex2
# Load Pretrained Model and Display the Annoted Results (multiple images)
from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolo26n.pt')

# Inference Source - multiple images
# Perform object detection on images using the model
results = model(['bus.jpg', 'bus2.jpg'])  # return a list of Results objects

# View results
for i, r in enumerate(results):
    # Plot results image
    dst = r.plot()      # return BGR-order numpy array
    cv2.imwrite(f'results_plot{i}.jpg', dst)
    cv2.imshow(f"result plot {i}", dst)

    # Save results to disk
    r.save(filename=f'results{i}.jpg')

cv2.waitKey(0)
cv2.destroyAllWindows()
```

### **Example 3: Inference on Webcam stream**

Create a new python source file in the project folder

* `Yolo26-Inference-Webcam-Ex3.py`

```python
# YOLO26 Tutorial : Prediction Ex3
# Stream Video Prediction
# This script will run predictions on each frame of the video,
# visualize the results, and display them in a window.
# The loop can be exited by pressing 'q'.

import cv2 as cv
from ultralytics import YOLO

# Load the YOLO26 model
model = YOLO('yolo26n.pt')
# Open the video camera no.0
cap = cv.VideoCapture(0)

# If not success, exit the program
if not cap.isOpened():
    print('Cannot open camera')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO26 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        # Display the annotated frame
        cv.imshow("YOLO26 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv.destroyAllWindows()
```

***

##

## Tutorial: Train YOLO26 with custom dataset

This tutorial is about learning how to train YOLO26 with a custom dataset of Mask-Dataset.

> This section is modified from the YOLOv8/YOLOv5 custom-data train tutorial

### Preparation:  Create Project Worspace

1. We will create the working space directory as

```
\DLIP\YOLOv26\
```

2. Then, create the sub-folder `/datasets` under the `/YOLOv26` folder

### Dataset: Custom Dataset

#### Download Dataset and Label

We will use the Labeled Mask YOLO dataset to detect people wearing mask.

![img](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1*kaURkIXnr0SfoxSVSdAEmg.png\&width=768\&dpr=3\&quality=100\&sign=3fc768f6\&sv=2)

This annotation file has 4 lines being each one referring to one specific face in the image. Let’s check the first line: The first integer number (0) is the object class id. For this dataset, the class id 0 refers to the class "using mask" and the class id 1 refers to the "without mask" class. The following float numbers are the `xywh` bounding box coordinates. As one can see, these coordinates are normalized to `[0, 1]`.



* Download the dataset : [Labeled Mask YOLO](https://www.kaggle.com/techzizou/labeled-mask-dataset-yolo-darknet).
* Under the directory `/YOLOv26/datasets`, create a new folder for the MASK dataset.&#x20;
* Then, copy the downloaded dataset under this folder.&#x20;
  * Example: `/YOLOv26/datasets/dataset_mask/archive/obj/`

![img](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fuser-images.githubusercontent.com%2F38373000%2F169465419-a354e40b-34e3-4608-b104-ae4f866f71a8.png\&width=768\&dpr=3\&quality=100\&sign=2abc663a\&sv=2)

The dataset is indeed a bunch of images and respective annotation files:

```
0 0.8024193548387096 0.5887096774193549 0.1596774193548387 0.2557603686635945
```

#### Visualize Train Dataset images

Under the working space (`YOLOv26/`), create the following python file (`visualizeLabel.py`) to view images and labels.

```python
## Visualize B.Box and Label on Train Dataset
import cv2

image_path = 'datasets/dataset_mask/archive/obj/2-with-mask'

image = cv2.imread(image_path + '.jpg')
if image is None:
    raise FileNotFoundError(image_path + '.jpg')

class_list = ['using mask', 'without mask']
colors = [(0, 255, 0), (0, 255, 255)]

height, width, _ = image.shape

T=[]
with open(image_path + '.txt', "r") as file1:
    for line in file1.readlines():
        split = line.split(" ")
        # getting the class id
        class_id = int(split[0])
        color = colors[class_id]
        clazz = class_list[class_id]

        # getting the xywh bounding box coordinates
        x, y, w, h = float(split[1]), float(split[2]), float(split[3]), float(split[4])
        # re-scaling xywh to the image size
        box = [int((x - 0.5*w)* width), int((y - 0.5*h) * height), int(w*width), int(h*height)]
        cv2.rectangle(image, box, color, 2)
        cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
        cv2.putText(image, class_list[class_id], (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

cv2.imshow("output", image)
cv2.waitKey()
```

You will see this result

![img](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fuser-images.githubusercontent.com%2F38373000%2F169467614-8907b0be-cda0-4f3d-9094-721c704e2886.png\&width=768\&dpr=3\&quality=100\&sign=87bf3831\&sv=2)

### Split Dataset

This step splits the original image-label pairs into training and validation sets.

YOLO uses the training set to learn object detection, and the validation set to check model performance during training.

* About 90% of the images and labels will be copied to the `training/` folders.
* The remaining 10% will be copied to the `validation/` folders.

The script will create the following structure:

* `datasets/dataset_mask/images/training/`
* `datasets/dataset_mask/images/validation/`
* `datasets/dataset_mask/labels/training/`
* `datasets/dataset_mask/labels/validation/`

For inference after training, you can use any test image containing people with or without masks.

Under the working directory create the following python file `split_data.py`.

{% code expandable="true" %}
```python
# Split Dataset as Train and Test
import os, shutil, random

# preparing the folder structure

full_data_path = 'datasets/dataset_mask/archive/obj/'
extension_allowed = '.jpg'
split_percentage = 90

images_path = 'datasets/dataset_mask/images/'
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)

labels_path = 'datasets/dataset_mask/labels/'
if os.path.exists(labels_path):
    shutil.rmtree(labels_path)
os.mkdir(labels_path)
training_images_path = images_path + 'training/'
validation_images_path = images_path + 'validation/'
training_labels_path = labels_path + 'training/'
validation_labels_path = labels_path +'validation/'

os.mkdir(training_images_path)
os.mkdir(validation_images_path)
os.mkdir(training_labels_path)
os.mkdir(validation_labels_path)

files = []

ext_len = len(extension_allowed)
for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            strip = file[0:len(file) - ext_len]
            files.append(strip)

random.shuffle(files)

size = len(files)

split = int(split_percentage * size / 100)

print("copying training data")
for i in range(split):
    strip = files[i]
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, training_images_path)

    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, training_labels_path)

print("copying validation data")
for i in range(split, size):
    strip = files[i]
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, validation_images_path)

    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, validation_labels_path)

print("finished")
```
{% endcode %}

Run the following script and check your project folders:

![Training and validation folder split result](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fykkimhgu%2FDLIP_doc%2Fassets%2F84508106%2Fd56a07b8-a671-4df9-85c7-4637532dc493\&width=768\&dpr=3\&quality=100\&sign=1eff9232\&sv=2)

### Create configuration file

The next step is to create the dataset configuration file, `maskdataset.yaml`, inside the `YOLOv26` directory.

This YAML file tells YOLO where the training and validation images are located and how many classes the dataset has.

```yaml
train: datasets/dataset_mask/images/training/
val: datasets/dataset_mask/images/validation/

# number of classes
nc: 2

# class names
names: ['with mask', 'without mask']
```

### Train

> You can increase the number of `epochs` for better training. `batch=-1` enables AutoBatch, which estimates a suitable batch size from available CUDA memory. It does not mean that GPU usage will always stay near 60%.

Create the following python file (`Yolo26_train.py`) to train model.

```python
from ultralytics import YOLO

def train():
    # Load a pretrained YOLO model
    model = YOLO('yolo26n.pt')

    # Train the model using the 'maskdataset.yaml' dataset for 3 epochs
    # batch=-1 estimates a suitable batch size from available CUDA memory
    results = model.train(data='maskdataset.yaml', epochs=3, batch=-1)

if __name__ == '__main__':
    train()
```

Finally, in the end, we have the following output:

![img](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fuser-images.githubusercontent.com%2F38373000%2F169474740-da09c0c0-a22e-4fc6-af0b-4d7dd28d5dd2.png\&width=768\&dpr=3\&quality=100\&sign=e987b3ec\&sv=2)

After training, check that the trained weight file `best.pt` was created.

The file is usually saved here:

`YOLOv26/runs/detect/train/weights/best.pt`

If you run training multiple times, Ultralytics creates a new folder such as `train-2`, `train-3`, and so on. In that case, check the latest training folder:

`YOLOv26/runs/detect/train-#/weights/best.pt`

![best.pt weight file location](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fykkimhgu%2FDLIP_doc%2Fassets%2F84508106%2Fd97809e0-608a-4c35-8095-64bca4b90e59.png\&width=768\&dpr=3\&quality=100\&sign=8fa8ae42\&sv=2)

Also, check the output of `runs/detect/train-#/results.png` which demonstrates the model performance indicators during the training:

![YOLO training results graph](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fmiro.medium.com%2Fmax%2F1400%2F1*Rxvpvfv7C7_HuPe9V-UmnA.png\&width=768\&dpr=3\&quality=100\&sign=f35d6d0\&sv=2)

### Inference

Now we have our model trained with the Labeled Mask dataset, it is time to get some predictions. This can be done using the same Ultralytics `YOLO(...).predict()` API:

Download a [test image here](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/mask-teens.jpg) and copy the file under the folder of `datasets/dataset_mask/images/testing`.

You can also download it directly from the Anaconda prompt:

```bat
if not exist datasets\dataset_mask\images\testing mkdir datasets\dataset_mask\images\testing
curl -L -o datasets\dataset_mask\images\testing\mask-teens.jpg https://raw.githubusercontent.com/ykkimhgu/DLIP-src/main/Tutorial_Pytorch/mask-teens.jpg
```

Create the following python file (`Yolo26_test.py`) to test model.

```python
from ultralytics import YOLO
import cv2

def test():
    # Load a pretrained YOLO model(Change model directory)
    model = YOLO('runs/detect/train/weights/best.pt')  # change this path if your result is train-2, train-3, ...

    # Inference Source - a single source(Change directory)
    src = cv2.imread("datasets/dataset_mask/images/testing/mask-teens.jpg")
    if src is None:
        raise FileNotFoundError("datasets/dataset_mask/images/testing/mask-teens.jpg")

    # Perform object detection on an image using the model
    result = model.predict(source=src, save=True, save_txt=True)  # save predictions as labels
    # View result
    for r in result:
        # print the Boxes object containing the detection bounding boxes
        print(r.boxes)

        # Plot results image
        print("result.plot()")
        dst = r.plot()  # return BGR-order numpy array
        cv2.imwrite("test_result_plot.jpg", dst)
        cv2.imshow("result plot", dst)

        # Plot the original image (NParray)
        print("result.orig_img")
        cv2.imwrite("test_result_orig.jpg", r.orig_img)
        cv2.imshow("result orig", r.orig_img)

        # Save results to disk
        r.save(filename='result.jpg')

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
```

Your result image will be saved under `runs/detect/predict/` or `runs/detect/predict-#/` if you run prediction multiple times.

![YOLO custom model inference result](https://ykkim.gitbook.io/dlip/~gitbook/image?url=https%3A%2F%2Fgithub.com%2Fykkimhgu%2FDLIP-src%2Fassets%2F84508106%2F07e98e4c-9e5d-48f5-9978-ec38ebaf1f0d\&width=768\&dpr=3\&quality=100\&sign=ffc1f421\&sv=2)

## Exercise

Test mask-detection trained YOLO with your webcam.

Create the following python file (`Yolo26_webcam.py`) to test the trained model with your webcam.

If your best model is saved under `train-2`, `train-3`, and so on, change the model path.

```python
from ultralytics import YOLO
import cv2

def webcam():
    # Load the trained YOLO model(Change model directory)
    model = YOLO('runs/detect/train/weights/best.pt')  # change this path if your result is train-2, train-3, ...

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the webcam frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO26 Webcam", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    webcam()
```

Run the script:

```bat
python Yolo26_webcam.py
```

Press `q` to close the webcam window.
