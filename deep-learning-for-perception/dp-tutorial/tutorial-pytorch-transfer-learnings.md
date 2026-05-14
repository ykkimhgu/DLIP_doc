---
description: updated 2026-5
---

# Tutorial: PyTorch Transfer Learning

## Tutorial: Transfer Learning using Pre-trained Models (Classification)

The purpose of this tutorial is to learn how to **transfer learning** using a pre-trained model.

In this document we will perform two types of **transfer learning**:

* **finetuning**: update all parameters of the pretrained model for our new task
* **feature extraction**: only update the final layer weights for predictions

## Preparation

First, you need to complete Tutorial: PyTorch Pretarin Model

* [Part1: inference using pre-trained model](tutorial-pytorch-pretrained.md#tutorial-inference-using-pre-trained-model-classification)

Also, refer to PyTorch tutorial:&#x20;

* [https://docs.pytorch.org/tutorials/beginner/transfer\_learning\_tutorial.html](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

We will download Python modules and image data.

*   [Download module](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Pytorch/TU_PyTorch_VSC/T3_2/initialize_model.py)

    > Move `initialize_model.py` to the **`models`** folder.

Create the main script

* **`TU_PyTorch_transfer_main.py`**

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

## Example: Transfer Learning from Resnet(imagenet)

### Model: Pretrained model

The classification models provided by torchvision are trained on ImageNet and consist of 1000 output layers.

Here, we want to fine-tune to other dataset with different class numbers.

* Use the `initialize_model()` module provided in the [pytorch tutorial](https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html) to change the output stage of the model.
* helps to initialize the fine-tuning of some models
* If the model is not in the function, the output layer information can be known by printing the model with the print() function.

#### Load ResNet with initialization\_model()

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

summary(model_ft, (3, input_size,input_size))

print(model_ft)
```

### Datasets: hymenoptera\_data

The downloaded datafile [hymenoptera\_data](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data)

* [hymenoptera\_data](https://www.kaggle.com/datasets/ajayrana/hymenoptera-data) is a binary (Ants and Bees) classification dataset consisting of a small number of images.

Unzip hymenoptera\_data.zip to create training data

* `hymenoptera_data.zip` should be in the subfolder `\data`

> Download and Move the `archive/hymenoptera_data/hymenoptera_data` to the **`data`** folder.

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

### Preprocessing: Train data

The images in the prepared dataset have different sizes. In order to be used as a learning model, the following process is required.

* Assign the images in the folder to training/test data for learning
* Same pre-processing as ImageNet data for input of learning model
* Resize the new dataset to the input size of pretrained model (e.g. 224 x 224)

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

### Optimization Setup

#### Optmizer function

Gradient descent is the common optimisation strategy used in neural networks. Many of the variants and advanced optimisation functions now are available,

* Stochastic Gradient Descent, Adagrade, Adam, etc

#### Loss function

* Linear regression: Mean Squared Error
* Classification: Cross entropy

```python
#########################################################
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
```

### Train: Transfer Learning with New Dataset

Modify Part 3 of the main script

```python
def train():
    # Run Train for k epoch
    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        training(train_dataloader, model_ft, loss_fn, optimizer, device, 15)
    print("Done!")

    # Save Train Model
    # * Need to create a new folder PATH priorly
    save_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    torch.save(model_ft, save_model_path)
```

### Inference:&#x20;

Modify Part 4 of the main script

{% code expandable="true" %}
```py
##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

def test():
    load_model_path = os.path.join(MODEL_DIR_PATH, MODEL_FILENAME)
    model_ft = torch.load(load_model_path, map_location=device, weights_only=False)

    evaluate(test_dataloader, model_ft, device)
```
{% endcode %}

### Visualize: test results

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

### Example code

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

summary(model_ft, (3, input_size,input_size))

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
optimizer = torch.optim.SGD(model_ft.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)

def train():
    # Run Train for k epoch
    for epoch in range(TOTAL_EPOCHS):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        training(train_dataloader, model_ft, loss_fn, optimizer, device, 15)
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

***

## Assignment

Apply Transfer Learng on  'EfficientNet(efficientnet\_b7)' by Training a Custom Dataset of Cat/Dog.

Then, show the test results on the test datasets.

### Model: EfficientNet <a href="#prepare-datasets-kaggle-cats-and-dogs" id="prepare-datasets-kaggle-cats-and-dogs"></a>

### Datasets: kaggle cats and dogs <a href="#prepare-datasets-kaggle-cats-and-dogs" id="prepare-datasets-kaggle-cats-and-dogs"></a>

Download the kaggle cats and dogs dataset: [download link](https://www.microsoft.com/en-us/download/details.aspx?id=54765)
