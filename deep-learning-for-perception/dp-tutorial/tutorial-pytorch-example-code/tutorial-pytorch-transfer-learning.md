---
description: updated 2026-5
---

# Tutorial: PyTorch Pretrain Model



## Tutorial: Inference using pre-trained model (classification)

classification model using a pretrained CNN model provided by PyTorch

The models were pre-trained on the **ImageNet** dataset (1000 classes)

## Preparation

1. Create the file
   * **`TU_PyTorch_pretrain_main.py`**

### Import Library

```python
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

import cv2 as cv
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image

# === Parameter === #
DATA_DIR_PATH = "data"
MODEL_DIR_PATH = "models"
MODEL_FILENAME = "vgg16_pretrained_model.pth"

BATCH_SIZE = 64
TOTAL_EPOCHS = 2
LEARNING_RATE = 1e-3
```

### GPU Setting

```python
##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if torch.cuda.is_available(): print(f'Device name: {torch.cuda.get_device_name(0)}') 
```

## Load a pre-trained model from TorchVision

Let’s import models from torchvision module and see what are the different models and architectures available with us.&#x20;

{% embed url="https://pytorch.org/vision/stable/models.html" %}

You can check the list by&#x20;

{% code expandable="true" %}
```py
dir(models)
```
{% endcode %}

Example:

* AlexNet is a class, whereas alexnet is a convenience function that returns the model instantiated from the AlexNet class.&#x20;
* densenet121, densenet161, densenet169, densenet201, all are instances of **DenseNet class** but with a different number of layers – 121, 161, 169 and 201, respectively.

### Load Pretrained VGG-16

We will use VGG-16 for this tutorial. Check the model architecture using summary

> Edit the **Part 2** in main file.

```python
##########################################################
## Part 2:  Load Pretrained Model
##########################################################

# Model Class Construction from Pretrained Model
model = models.vgg16(weights='DEFAULT')
model.eval() # run the model with evaluation mode
model = model.to(device)

summary(model, (3, 224, 224))
```

## Test image preparation

### Download Test Image

(Option 1)

In this tutorial, we load one test image file: [download image here](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/TU_PyTorch_VSC/test_image_dog.png)

* Save the image file under `data\` folder&#x20;



(Option 2)

You can also use the test image file from URL

> Create the **Part 1** in main file.

```python
##########################################################
## Part 1:  Prepare Image and Label
##########################################################

filename = os.path.join(DATA_DIR_PATH, "test_image_dog.png")

##(Option 2)
url = "https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png"
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# image show
img = cv.imread(filename)
dst = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(dst)
plt.show()
```

### Download  Label data

We need to get the class label list.&#x20;

Thus, we will load label information from a text file having a list of all the 1000 class labels. The line number specifies the class number

```python
# Download ImageNet labels
label_filename = os.path.join(DATA_DIR_PATH, 'imagenet_classes.txt')

url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
try: urllib.request.urlretrieve(url, label_filename)
except: urllib.request.urlretrieve(url, label_filename)
```

### Preprocessing Test Data

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224.&#x20;

The images have to be loaded in to a range of \[0, 1] and then normalized using mean = \[0.485, 0.456, 0.406] and std = \[0.229, 0.224, 0.225]  (ImageNet dataset).

> Add it to **Part 1** in main file.

```
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```



## Inference using pretrained model

The output is the probability value for each 1000 classes. (the sum of all probabilities is 1)

> Create the **def visualize()** in main file.

```python
##########################################################
## Part 5:  Visualize Result
##########################################################

def visualize():
    # sample execution (requires torchvision)
    # Normalize and resize to 224x224
    input_image = Image.open(filename)
    input_tensor = data_transform(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device) # create a mini-batch as expected by the model

    # Forward process
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)
```



Now, we need to find out the index for the maximum probability. This index is the prediction class. For this tutorial, we will print the top-5 probability

> Add it to **def visualize()** in main file.

```python
# Read the categories
with open(label_filename, "r") as f:
    categories = [s.strip() for s in f.readlines()]

# Show top 5 categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)

for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())
```

## Example: Main Script

{% code expandable="true" %}
```python
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

import cv2 as cv
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image


# === Parameter === #
DATA_DIR_PATH = "data"
MODEL_DIR_PATH = "models"
MODEL_FILENAME = "vgg16_pretrained_model.pth"

BATCH_SIZE = 64
TOTAL_EPOCHS = 2
LEARNING_RATE = 1e-3


##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")
if torch.cuda.is_available(): print(f'Device name: {torch.cuda.get_device_name(0)}') 


##########################################################
## Part 1:  Prepare Dataset 
##########################################################

filename = os.path.join(DATA_DIR_PATH, "test_image_dog.png")

##(Option 2)
url = "https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png"
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# image show
img = cv.imread(filename)
dst = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(dst)
plt.show()

# Download ImageNet labels
label_filename = os.path.join(DATA_DIR_PATH, 'imagenet_classes.txt')

url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
try: urllib.request.urlretrieve(url, label_filename)
except: urllib.request.urlretrieve(url, label_filename)


# transformation to tensor:  converts 0~255 value to 0~1 value.
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



##########################################################
## Part 2:  Create Model Instance 
##########################################################

# Model Class Construction
model = models.vgg16(weights='DEFAULT')
model.eval() # run the model with evaluation mode
model = model.to(device)

summary(model, (3, 224, 224))



##########################################################
## Part 5:  Visualize Result
##########################################################

def visualize():
    # sample execution (requires torchvision)
    # Normalize and resize to 224x224
    input_image = Image.open(filename)
    input_tensor = data_transform(input_image)
    input_batch = input_tensor.unsqueeze(0).to(device) # create a mini-batch as expected by the model

    # Forward process
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    #print(output[0])

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print(probabilities)


    # Read the categories
    with open(label_filename, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # Show top 5 categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())



##########################################################
## MAIN
##########################################################
if __name__ == "__main__":
    visualize()
```
{% endcode %}


