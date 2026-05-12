---
description: updated 2026-5
---

# Tutorial: PyTorch Transfer Learning

* T3-1: [Test using Pretrained Model (VGG, Inception, ResNet)](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_1_Inference_using_Pre_trained_Model_\(classification\)_2024.ipynb)

* T3-2: [Transfer Learning of Pretrained model](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/Tutorial_PyTorch_T3_2_Transfer_Learning_using_Pre_trained_Models_\(classification\)_2024.ipynb)





## Part 1: Inference using pre-trained model (classification)

classification model using a pretrained CNN model provided by PyTorch

The models were pre-trained on the **ImageNet** dataset (1000 classes)



## Preparation

1. Create the file
   * `**T3_1_main.py**`

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

Let’s import models from torchvision module and see what are the different models and architectures available with us. (see: https://pytorch.org/vision/stable/models.html)

Notice that there is one entry called AlexNet and one called alexnet. The capitalised name refers to the Python class (AlexNet) whereas alexnet is a convenience function that returns the model instantiated from the AlexNet class. These convenience functions can have different parameter sets.

Densenet121, densenet161, densenet169, densenet201, all are instances of DenseNet class but with a different number of layers – 121,161,169 and 201, respectively.

### Load Pretrained VGG-16

We will use VGG-16 for this tutorial. Check the model architecture using summary

> Edit the **Part 2** in main file.

```python
##########################################################
## Part 2:  Load Pretrained Model
##########################################################

# Model Class Construction
model = models.vgg16(weights='DEFAULT')
model.eval() # run the model with evaluation mode
model = model.to(device)

summary(model, (3, 224, 224))
```



## Test image preparation

In this tutorial, we load one test image file from the following URL

>  Edit the **Part 1** in main file.

```python
##########################################################
## Part 1:  Prepare Image
##########################################################

url = "https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png"
filename = os.path.join(DATA_DIR_PATH, "test_image.jpg")

try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# image show
img = cv.imread(filename)
dst = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(dst)
plt.show()
```



## Inference using pretrained model

All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

> Add it to **Part 1** in main file.

```
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
```

Here's a sample execution.

The output is the probability value for each 1000 classes. (the sum of all probabilities is 1)

> Edit the **def visualize()** in main file.

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

What do we do with the output which is a vector with 1000 elements? We need to get class label list of the image.

Thus, we will load label information from a text file having a list of all the 1000 class labels. The line number specifies the class number

> Add it to **def visualize()** in main file

```python
# Download ImageNet labels
url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
label_filename = os.path.join(DATA_DIR_PATH, 'imagenet_classes.txt')
urllib.request.urlretrieve(url, label_filename)
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

{% tabs %}
{% tab title="main.py" %}
{% code expandable="true" %}

````python
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
...



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

url = "https://3.bp.blogspot.com/-W__wiaHUjwI/Vt3Grd8df0I/AAAAAAAAA78/7xqUNj8ujtY/s1600/image02.png"
filename = os.path.join(DATA_DIR_PATH, "test_image.jpg")

try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# image show
img = cv.imread(filename)
dst = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(dst)
plt.show()

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
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

def train():
    ...


##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

def test():
    ...


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

    # Download ImageNet labels
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    label_filename = os.path.join(DATA_DIR_PATH, 'imagenet_classes.txt')
    urllib.request.urlretrieve(url, label_filename)

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
    # train()
    # test()
    visualize()
````

{% endcode %}
{% endtab %}





## Part 2: Transfer Learning using Pre-trained Models (Classification)

- Part1: inference using pre-trained model
- **Part2: Transfer Learning using Pre-trained Models (Classification)**

The purpose of this tutorial is to learn how to **transfer learning** using a pre-trained model.

In this document we will perform two types of **transfer learning**:

- **finetuning**: update all parameters of the pretrained model for our new task
- **feature extraction**: only update the final layer weights for predictions

## Preparation

1. we will download Python modules and image data.

- [download modules](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/2024/T3_2_pytorch_classification_modules.zip)

  > Move `initialize_model.py` and `set_parameter_requires_grad.py` to the `**models**` folder.

- [download dataset(ant/bee)](https://drive.google.com/file/d/123qUnqUpSzpnj7BnJjftFClmK6PLRzfA/view?usp=sharing)

  > Move to the `**data**` folder.

2. Create the file
   * `**T3_2_main.py**`

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.train import train as training
from utils.eval import evaluate
```

### Import from the downloaded modules

```python
from models.initialize_model import initialize_model
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



## Load Pretrained MODEL

Basically, the classification models provided by torchvision are trained on ImageNet and consist of 1000 output layers.

However, in the model for fine-tuning with other datasets, the number of output layers should be different depending on the class.

Here, we use the initialize_model() function provided in the [pytorch tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) to change the output stage of the model.

initialize_model() is a function that helps to initialize the fine-tuning of some models.

If the model is not in the function, the output layer information can be known by printing the model with the print() function.

### Load ResNet with initialization_model()

```python
##########################################################
## Part 1:  Create Model Instance 
##########################################################

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception*]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 2

# - True(feature extraction): only update the reshaped layer params,
# - False(finetuning)       : finetune the whole model, 
feature_extract = True  

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

summary(model_ft, (3,input_size,input_size))

print(model_ft)
```



## Prepare Datasets: hymenoptera_data

The downloaded datafile `hymenoptera_data.zip` should be in the subfolder `\data`

Unzip hymenoptera_data.zip to create training data

[hymenoptera_data](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data) is a binary (Ants and Bees) classification dataset consisting of a small number of images.

> Download and Move the `archive/hymenoptera_data/hymenoptera_data` to the `**data**` folder.

```python
# === Parameter === #
DATA_DIR_PATH = "data/hymenoptera_data"
MODEL_DIR_PATH = "models"
MODEL_FILENAME = "ResNet_ft(hymenoptera).pth"

# Batch size for training (change depending on how much memory you have)
BATCH_SIZE = 8
TOTAL_EPOCHS = 2
LEARNING_RATE = 1e-3
```

The images in the prepared dataset have different sizes. In order to be used as a learning model, the following process is required.

- Assign the images in the folder to training/test data for learning
- Same pre-processing as ImageNet data for input of learning model
- Resize the new dataset to the input size of pretrained model (e.g. 224 x 224)

```python
##########################################################
## Part 2:  Prepare Dataset 
##########################################################

# Data augmentation and normalization for training
# Just normalization for validation
# Normalized with ImageNet mean and variance
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

training_data = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR_PATH, 'train'), 
    transform=data_transform['train'],
    )

test_data = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR_PATH, 'val'), 
    transform=data_transform['val'],
    )

classes = ['ant', 'bee']
print(f"train dataset length = {len(training_data)}")
print(f"test  dataset length = {len(test_data)}")
```

Use DataLoader to make dataset iterable.

```python
train_dataloader = torch.utils.data.DataLoader(
    training_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    )

test_dataloader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    )

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape} {y.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```



## Optimization Setup

### Optmizer function

Gradient descent is the common optimisation strategy used in neural networks. Many of the variants and advanced optimisation functions now are available,

- Stochastic Gradient Descent, Adagrade, Adam, etc

### Loss function

- Linear regression: Mean Squared Error
- Classification: Cross entropy

```python
#########################################################
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9,weight_decay=5e-4)
```



## Transfer Learning with New Dataset

```python
def train():
    # Run Train for k epoch
    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        training(train_dataloader, model_ft, loss_fn, optimizer, device)
    print("Done!")

    # Save Train Model
    # * Need to create a new folder PATH priorly
    save_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    torch.save(model_ft, save_model_path)


##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

def test():
    load_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    model_ft = torch.load(load_model_path, map_location=device, weights_only=False)

    evaluate(test_dataloader, model_ft, device)
```



## Visualize test results

Select random test images and evaluate

```python
##########################################################
## Part 5:  Visualize Evaluation Results
##########################################################

def visualize():
    # # Get some random test  images // BatchSize at a time
    dataiter = iter(test_dataloader)
    images, labels = next(dataiter)
    print(images.size())


    # Evaluate mode
    # Prediction of some sample images 
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        pred = model_ft(images)
        _, predicted = torch.max(pred.data, 1)


    # Plot BATCH_SIZE-9 output images
    figure = plt.figure()
    num_of_images = min(BATCH_SIZE, 9)

    for index in range(num_of_images):
        plt.subplot(3, 3, index+1)
        plt.axis('off')
        # plt.title(f"Ground Truth: {classes[labels[index]]}")  
        plt.title(f"Predicted: {classes[predicted[index].item()]}")
        # 출력을 위한 차원변환 (channels*rows*cols) -> (rows*cols*channels)
        plt.imshow(np.transpose((images[index] * 0.224  + 0.456).cpu().numpy().squeeze(), (1, 2, 0)))
    plt.show()
```



Plot heatmap (confusion matrix)

> Add it to `def visualize()`

```python
    # Get some random test  images // BatchSize at a time
    heatmap = pd.DataFrame(data=0, index=classes, columns=classes)

    for images, labels in test_dataloader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                heatmap.iloc[true_label, predicted_label] += 1
    print(heatmap)
    _, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu")
    plt.show()
```



## Example: Main Script

{% tabs %}
{% tab title="main.py" %}
{% code expandable="true" %}

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from models.initialize_model import initialize_model
from utils.train import train as training
from utils.eval import evaluate


# === Parameter === #
DATA_DIR_PATH = "data/hymenoptera_data"
MODEL_DIR_PATH = "models"
MODEL_FILENAME = "ResNet_ft(hymenoptera).pth"

BATCH_SIZE = 8
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
## Part 1:  Create Model Instance 
##########################################################

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception*]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 2

# - True(feature extraction): only update the reshaped layer params,
# - False(finetuning)       : finetune the whole model, 
feature_extract = True  

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
model_ft = model_ft.to(device)

summary(model_ft, (3,input_size,input_size))

print(model_ft)


##########################################################
## Part 2:  Prepare Dataset 
##########################################################

# Data augmentation and normalization for training
# Just normalization for validation
# Normalized with ImageNet mean and variance
data_transform = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

training_data = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR_PATH, 'train'), 
    transform=data_transform['train'],
    )

test_data = torchvision.datasets.ImageFolder(
    root=os.path.join(DATA_DIR_PATH, 'val'), 
    transform=data_transform['val'],
    )


classes = ['ant', 'bee']
print(f"train dataset length = {len(training_data)}")
print(f"test  dataset length = {len(test_data)}")


train_dataloader = torch.utils.data.DataLoader(
    training_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    )

test_dataloader = torch.utils.data.DataLoader(
    test_data, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    )

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape} {y.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


#########################################################
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9,weight_decay=5e-4)

def train():
    # Run Train for k epoch
    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        training(train_dataloader, model_ft, loss_fn, optimizer, device)
    print("Done!")

    # Save Train Model
    # * Need to create a new folder PATH priorly
    save_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    torch.save(model_ft, save_model_path)


##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

def test():
    load_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    model_ft = torch.load(load_model_path, map_location=device, weights_only=False)

    evaluate(test_dataloader, model_ft, device)


##########################################################
## Part 5:  Visualize Evaluation Results
##########################################################

def visualize():
    # # Get some random test  images // BatchSize at a time
    dataiter = iter(test_dataloader)
    images, labels = next(dataiter)
    print(images.size())


    # Evaluate mode
    # Prediction of some sample images 
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        pred = model_ft(images)
        _, predicted = torch.max(pred.data, 1)


    # Plot BATCH_SIZE-9 output images
    figure = plt.figure()
    num_of_images = min(BATCH_SIZE, 9)

    for index in range(num_of_images):
        plt.subplot(3, 3, index+1)
        plt.axis('off')
        # plt.title(f"Ground Truth: {classes[labels[index]]}")  
        plt.title(f"Predicted: {classes[predicted[index].item()]}")
        # 출력을 위한 차원변환 (channels*rows*cols) -> (rows*cols*channels)
        plt.imshow(np.transpose((images[index] * 0.224  + 0.456).cpu().numpy().squeeze(), (1, 2, 0)))
    plt.show()

    # Get some random test  images // BatchSize at a time
    heatmap = pd.DataFrame(data=0, index=classes, columns=classes)

    for images, labels in test_dataloader:
        with torch.no_grad():
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = predicted[i].item()
                heatmap.iloc[true_label, predicted_label] += 1
    print(heatmap)
    _, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu")
    plt.show()


##########################################################
## MAIN
##########################################################
if __name__ == "__main__":
    train()
    test()
    visualize()
```

{% endcode %}
{% endtab %}
