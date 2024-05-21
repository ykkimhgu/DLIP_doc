# Tutorial: Yolov8 in PyTorch

## Tutorial: YOLO v8 in PyTorch

https://docs.ultralytics.com/quickstart/#install-ultralytics

[Ultralytics](https://ultralytics.com/) [YOLOv8](https://github.com/ultralytics/ultralytics) is a cutting-edge, state-of-the-art (SOTA) model that builds upon the success of previous YOLO versions and introduces new features and improvements to further boost performance and flexibility. Y

![img](https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png)

### Documentation and Github

See the [YOLOv8 Docs](https://docs.ultralytics.com/) for full documentation on training, validation, prediction and deployment.

Also, you can visit the github repository: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

***

##

## Installation of Yolov8

Ultralytics provides various installation methods including pip, conda, and Docker. Install YOLOv8 via the `ultralytics` pip package for the latest stable release or by cloning the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics) for the most up-to-date version.

### Requirement

Install necessary packages such as Python, Numpy, PyTorch, CUDA and more

For installations of requirements, [read for more detail instructions](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning#part-3.-installing-dl-framework)

* Python >=3.8
* PyTorch>=1.8
* opencv-python>=4.6.0
* matplotlib>=3.3.0
* and more. [See requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)

### Install Yolov8 via pip package

First, create a new environment for YOLOv8 in Anaconda Prompt.

* e.g. $myENV$ = yolov8

You can also make an exact copy of the existing environment by creating a clone

* If you already have an environment named `py39`, clone it as `yolov8`

```bash
conda create --name yolov8 --clone py39

```

Activate the environment and Install YOLOv8 with pip to get stable packages.

Also, install the latest ONNX

```bash
conda activate yolov8
pip install ultralytics
pip install onnx
```

### Check for YOLO Installation

After the installation, you can check the saved source code and libs of YOLOv8 in the local folder :

`\USER\anaconda3\envs\yolov8\Lib\site-packages\ultralytics`

Now, lets run simple prediction examples to check the YOLO installation.

In Anaconda Prompt, activate `yolov8` environment.

Then, move directory to the working directory. Here, the result images will be saved.

* Example: `C:\Users\ykkim\source\repos\DLIP\yolov8\`

```powershell
conda activate yolov8

cd C:\Users\ykkim\source\repos\DLIP\yolov8
```

#### Run a Detection Example

In the Anaconda prompt, type the following command to predict a simple image.

```cmd
yolo predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
```

The result will be saved in the project folder `\runs\detect\predict\`

> Example: C:\Users\ykkim\source\repos\DLIP\yolov8\runs\detect\predict\\

<figure><img src="../../../.gitbook/assets/image (1).png" alt="" width="375"><figcaption></figcaption></figure>

#### Run a Segmentation Example

Predict a YouTube video using a pretrained segmentation model at image size 320:

```cmd
yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
```

The result will be saved in the project folder `\runs\segment\predict\`

<figure><img src="../../../.gitbook/assets/image (2).png" alt="" width="375"><figcaption></figcaption></figure>

***

## Using YOLOv8 with Python : Example Codes

In the project folder, create a new python code file

* Project Folder: \source\repos\DLIP\yolov8\\
* Activate `yolov8` environment in Anaconda Prompt

A list of useful commands for YOLOv8

```python
from ultralytics import YOLO

# Create a new YOLO model from scratch
model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(data='coco8.yaml', epochs=3)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
results = model('https://ultralytics.com/images/bus.jpg')


```

### Example: Detection Inference

Read [Doc of Prediction with YOLO for more examples](https://docs.ultralytics.com/modes/predict/)

Download the dataset file and save in the project folder

* [bus.jpg](https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg)

You can download the COCO pretrained models such as YOLOv8n and more.

[https://docs.ultralytics.com/datasets/detect/coco/](https://docs.ultralytics.com/datasets/detect/coco/)

#### Inference one image

Create a new python source file in the project folder

* Yolo8-Inference-Ex1.py

```python
#########################################################
# YOLO v8  Tutorial : Prediction  Ex1
#
# Load Pretrained Model and Display the Annoted Results
#
#########################################################

from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt



# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Inference Source - a single source
src = cv2.imread("bus.jpg")


# Perform object detection on an image using the model
result = model.predict(source=src, save=True, save_txt=True)  # save predictions as labels


# View result   
for r in result:

    # print the Boxes object containing the detection bounding boxes        
    print(r.boxes)  
    
    # Show results to screen (not recommended)
    print("result.show()")
    r.show()    

   
    # Plot results image    
    print("result.plot()") 
    dst = r.plot()      # return BGR-order numpy array
    cv2.imshow("result plot",dst)         
    

    # Plot the original image (NParray)
    print("result.orig_img")
    cv2.imshow("result orig",r.orig_img)   


# Save results to disk
r.save(filename='result.jpg')
cv2.waitKey(0)



##########################################################################################



```

#### Inference of multiple images

Create a new python source file in the project folder

* Yolo8-Inference-Ex2.py

For multiple input source images, you can copy `bus.jpg` as `bus2.jpg`.

```python
#########################################################
# YOLO v8  Tutorial : Prediction  Ex2
#
# Load Pretrained Model and Display the Annoted Results (multiple images)
#
#########################################################


from ultralytics import YOLO
import cv2
from matplotlib import pyplot as plt


# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')


# Inference Source - multiple images
# Perform object detection on images using the model
results = model(['bus.jpg', 'bus2.jpg'])  # return a list of Results objects


# View results
for i, r in enumerate(results):

    # Plot results image    
    dst = r.plot()      # return BGR-order numpy array
    cv2.imshow("r.plot",dst)       
   

    # Save results to disk
    r.save(filename=f'results{i}.jpg')
    
cv2.waitKey(0)


```

#### Inference on Webcam stream

Create a new python source file in the project folder

* Yolo8-Inference-Webcam-Ex3.py

```python
#########################################################
# YOLO v8  Tutorial : Prediction  Ex3
#
# Stream Video Prediction 
#
# This script will run predictions on each frame of the video
# visualize the results, and display them in a window. 
# The loop can be exited by pressing 'q'.
#########################################################


import cv2 as cv
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

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
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv.imshow("YOLOv8 Inference", annotated_frame)

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

### Example: Train
