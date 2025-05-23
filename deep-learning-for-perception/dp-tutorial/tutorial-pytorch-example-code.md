# Tutorial: PyTorch Example Code

## Preparation

1. Create the working directory:  such as **`\TU_MLP_CNN_Classification\`**
2. Download the Tutorial Code in Zip file:  [TU\_MLP\_CNN\_Classification\_Example Code 2024.zip](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode/TU\_MLP\_CNN\_Classification\_Example%20Code%202024.zip)
3. Unzip the files in the working directory.&#x20;
4. Make sure you have the subdirectory for saving weights:  **`\TU_MLP_CNN_Classification\weights\`**



## Example 1. ANN(MLP) : Model, Train, Test

Classify the MNIST digit with a simple ANN\\

<figure><img src="https://camo.githubusercontent.com/bd39a5ca7bbfdc90303a170fdd99c9faaeff98a80ccb45f5fb96aa8f7d7ebe5a/68747470733a2f2f6769746875622e636f6d2f62656e747265766574742f7079746f7263682d696d6167652d636c617373696669636174696f6e2f626c6f622f6d61737465722f6173736574732f6d6c702d6d6e6973742e706e673f7261773d31" alt=""><figcaption></figcaption></figure>

* Image Input: 1x28x28 image
* Flatten into a 1x28\*28 element vector
* 1st Layer: linear to 250 dimensions / ReLU
* 2nd Layer: linear to 100 dimensions / ReLU
* 3rd Layer: linear to 10 dimensions / log SoftMax
* Output: 1x10
* Activation function: ReLU

### Creating  Model Architecture, Training and Evaluation&#x20;

We will create a class for the  model architecture

Also, we will create PyTorch source files  for processing (1) Training and (2) Evaluation of the model

1. Check if you have created the subfolder **`.\weights\`**
2. Download the  source code for **Model Architecture**: [myModel.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
3. Download the source code for **Model Training**: [TU\_PyTorch\_MLP\_Train.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
   * Run this code and observe the loss
   * Change the training epoch
4. Download the source code for **Model Evaluation**: [TU\_PyTorch\_MLP\_Eval.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
   * Run this code and observe the evaluation accuracy and sample images

{% tabs %}
{% tab title="myModel.py" %}
````python
```python
##########################################################
# PyTorch Tutorial:  MLP & CNN Model Architecture
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: User defined ANN & CNN Model & Modules for Training and Testing
#
# Model: MLP, LeNet
#
##########################################################

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt




##########################################################
##  Model Architecture   
##########################################################


# Model Architecture: MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

        
    def forward(self, x):
        x=self.flatten(x)
        x= F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = F.log_softmax(self.linear3(x))
        return y_pred


     

##########################################################
##  Train Module
##########################################################

# Train Module
def train(dataloader, model, loss_fn,optimizer, device, print_interval=100):
    # Dataset Size
    size = len(dataloader.dataset)
    
    # Batch size
    batch_size = dataloader.batch_size
    print(f"batch size : {batch_size}")

    # Model in Training Mode
    model.train()

    running_loss=0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # zero gradients for every batch
        optimizer.zero_grad()   

        # Compute prediction loss 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and Update        
        loss.backward()
        optimizer.step()        

        # Print avg. loss for every mini-batch in an epoch
        running_loss+=loss.item()
        if batch % print_interval == 0:
            running_loss=running_loss/print_interval            
            current = batch * batch_size
            print(f"loss: {running_loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss=0




##########################################################
##  Test Model Module 
##########################################################

def test(dataloader, model, device):
    # Dataset Size
    size = len(dataloader.dataset)

    # Batch Size
    num_batches = len(dataloader)
    
    # Model in Evaluation Mode
    model.eval()

    #test_loss=0
    correctN = 0
    
    # Disable grad() computation to reduce memory consumption.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Compute average prediction loss 
            pred = model(X)            
            #test_loss += loss_fn(pred, y).item()

            # Predict Label
            y_pred=pred.argmax(1);
            correctN += (y_pred == y).type(torch.float).sum().item()
            
    #test_loss /= num_batches
    correctN /= size
    #print(f"Test Error: \n Accuracy: {(100*correctN):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100*correctN):>0.1f}% \n")


```
````
{% endtab %}

{% tab title="TU_PyTorch_MLP_Train" %}
````python
```python
##########################################################
# PyTorch Tutorial:  Classification MLP Model for Training
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: This example is creating and training a MLP model for classification
#
# Model: MLP
# Dataset: MNIST
#
##########################################################


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt

import myModel  as myModel # User defined modules

PATH='./weights/'

##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



##########################################################
## Part 1:  Prepare Dataset 
##########################################################

# Download Dataset from TorchVision MNIST
# Once, downloaded locally, it does not download again.
#
#  MNIST shape:  1x28x28 

# transformation to tensor:  converts 0~255 value to 0~1 value.
data_transform = transforms.Compose([            
            transforms.ToTensor(),
])

# TRAIN DATA
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=data_transform,   
)


# Create DataLoader with Batch size N
# Input Dim:  [N, C, H, W]
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break




##########################################################
## Part 2:  Create Model Instance 
##########################################################

# Model Class Construction
model = myModel.MLP().to(device)
print(model)



##########################################################
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Run Train for k epoch
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    myModel.train(train_dataloader, model,loss_fn, optimizer,device)    
print("Done!")


# Save Train Model
# * Need to create a new folder PATH priorly
torch.save(model,PATH+'MLP_MNIST_model.pth')


# Evaluate the trained model in **Eval.py

```
````
{% endtab %}

{% tab title="TU_PyTorch_MLP_Eval" %}
````python
```python
##########################################################
# PyTorch Tutorial:  Classification CNN Model for Evaluation
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: This example is tesing pretrined CNN model for classification
#
# Model: LeNet
# Dataset: MNIST
#
##########################################################


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt

# User defined Model
import myModel as myModel

# Model weight directory
ModelPATH='./weights/'



##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



##########################################################
## Part 1:  Prepare Dataset
##########################################################

# Download Dataset from TorchVision MNIST
# Once, downloaded locally, it does not download again.
#
#  MNIST shape:  1x28x28 

# transformation to tensor:  converts 0~255 value to 0~1 value.
data_transform = transforms.Compose([            
            transforms.ToTensor(),
])


# EVAL DATA
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=data_transform,
)


# Create DataLoader with Batch size N
# Input Dim:  [N, C, H, W]
batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

model=torch.load(ModelPATH+'MLP_MNIST_model.pth')
myModel.test(test_dataloader, model, device)




##########################################################
## Part 5:  Visualize Evaluation Results
##########################################################

# Select one batch of images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)
print(images.size())


# Evaluate on one batch test images
images, labels = images.to(device), labels.to(device)
with torch.no_grad():
    pred = model(images)
    _, predicted = torch.max(pred.data, 1)


# Plot 9 output images
figure = plt.figure()
num_of_images = 9
for index in range(num_of_images):
    plt.subplot(3, 3, index+1)
    plt.axis('off')    
    plt.title("Predicted: {}".format(predicted[index].item()))
    plt.imshow(images[index].cpu().numpy().squeeze(), cmap='gray_r')
plt.show()
```
````
{% endtab %}
{% endtabs %}

###

### (For CoLAB) Everything in one source file

<details>

<summary>(For CoLAB) Everything in one source file</summary>

````python
```python
##########################################################
# PyTorch Tutorial:  Classification MLP Model for Evaluation
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: This example is tesing pretrined model for classification
#
# Model:   MLP
# Dataset: MNIST
#
##########################################################
```

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


##########################################################
## Part 1:  Prepare Dataset
##########################################################

# Download Dataset from TorchVision MNIST
# Once, downloaded locally, it does not download again.
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),   #converts 0~255 value to 0~1 value.
)
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

# Create DataLoader with Batch size N
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break


##########################################################
## Part 2:  Create Model
##########################################################

# Model Architecture
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(28*28, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

    def forward(self, x):
        x=self.flatten(x)
        x= F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = F.log_softmax(self.linear3(x))
        return y_pred

# Model Class
model = MLP().to(device)
print(model)

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)


##########################################################
## Part 3:  Train Model
##########################################################

# Train Module
def train(dataloader, model, loss_fn, optimizer):
    # Dataset Size
    size = len(dataloader.dataset)
    
    # Model in Training Mode
    model.train()

    running_loss=0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # zero gradients for every batch
        optimizer.zero_grad()   

        # Compute prediction loss 
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation and Update        
        loss.backward()
        optimizer.step()        

        # Print loss for every 100 batch in an epoch
        running_loss+=loss.item()
        if batch % 100 == 0:
            running_loss=running_loss/100
            current = batch * len(X)
            print(f"loss: {running_loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss=0

# Run Train for k epoch
epochs = 2
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)    
print("Done!")

# Save Train Model
torch.save(model,"MNIST_model.pth")


##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

def test(dataloader, model, loss_fn):
    # Dataset Size
    size = len(dataloader.dataset)

    # Batch Size
    num_batches = len(dataloader)
    
    # Model in Evaluation Mode
    model.eval()

    test_loss, correctN = 0, 0
    
    # Disable grad() computation to reduce memory consumption.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Compute average prediction loss 
            pred = model(X)            
            test_loss += loss_fn(pred, y).item()

            # Predict Label
            y_pred=pred.argmax(1);
            correctN += (y_pred == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correctN /= size
    print(f"Test Error: \n Accuracy: {(100*correctN):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Run Eval for k epoch
test(test_dataloader, model, loss_fn)


##########################################################
## Part 5:  Visualize Evaluation Results
##########################################################

# Select one batch of images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)
print(images.size())

# Evaluate on one batch test images
with torch.no_grad():
  for X, y in dataiter:
      X, y = X.to(device), y.to(device)
      
      # Prediction Label 
      pred = model(X)
      _, predicted = torch.max(pred.data, 1)

# Plot 
figure = plt.figure()
num_of_images = 9
for index in range(num_of_images):
    plt.subplot(3, 3, index+1)
    plt.axis('off')    
    plt.title("Predicted: {}".format(predicted[index].item()))
    plt.imshow(images[index].cpu().numpy().squeeze(), cmap='gray_r')
    plt.show()


````

</details>

###

## Example 2. CNN : Model, Train, Test

LeNet-5 Architecture. Credit: [LeCun et al., 1998](http://yann.lecun.com/exdb/publis/psgz/lecun-98.ps.gz)

<figure><img src="https://camo.githubusercontent.com/1e36248a674f011bd232d205c1d10f4e09976c22410452d7a5086058b00f4599/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f4d6f737461666147617a61722f6d6f62696c652d6d6c2f6d61737465722f66696c65732f6c656e65742e706e67" alt=""><figcaption></figcaption></figure>

**Architecture**

* **\[C1] Conv:** Input (32x32x1) to Output (28x28x6) by 5x5 filters, `relu`
* **\[S2] Pooling** : Input (28x28x6) to Output (14x14x6), `maxPooling 2x2`
* **\[C3] Conv:** Input (14x14x6) to Output (10x10x16) by 5x5 filters, `relu`
* **\[S4] Pooling** : Input (10x10x16) to Output (5x5x16), `maxPooling 2x2`
* **Flatten** : Input (5x5x16) to Output 1D 1x (5 \* 5 \* 16)
* **\[F5] FC** : Input (1x5 \* 5 \* 16) to Output (1x120) , `relu`
* **\[F6] FC** : Input (1x120) to Output (1x84) , `relu`
* **\[OUTPUT]** : Input (1x84) to Output (1x10)

####

*

### Creating  Model Architecture, Training and Evaluation&#x20;



We will worki  in the same working directory as Example 1:  e.g.  **`\TU_MLP_CNN_Classification\`**



We will create a class for the  model architecture. Also, we will create PyTorch source files  for processing (1) Training and (2) Evaluation of the model

1. Update `myModel.py` by adding   `class LeNet5`&#x20;
2. Download the source code for **Model Training**: [TU\_PyTorch\_CNN\_Train.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
   * Run this code and observe the loss
   * Change the training epoch
3. Download the source code for **Model Evaluation**: [TU\_PyTorch\_CNN\_Eval.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
   * Run this code and observe the evaluation accuracy and sample images

### Other Style using nn.Sequencial()

**Important Note**

* in Sequential(). use nn.ReLU() , NOT F.relu()
* ( nn.ReLU() vs F.relu() ) Watch out for spelling and Captital Letters
* add comma `,` in nn.Sequential(). Such as nn.Conv2d(1,6,5), nn.ReLU() etc

{% tabs %}
{% tab title="myModel.py" %}


```python
# myModel.py

# ... same as Example 1

# Model Architecture
class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.flatten = nn.Flatten()

        # Feature Extraction 
        # Conv2d(input ch, output ch, convolution,stride=1)
        # (1,6,5) = Input ch=1, Output ch=6 with 5x5 conv kernel        
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
                
        # Classifier
        self.fc1 = nn.Linear(16*5*5,120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):        
        # C1
        x=self.conv1(x)
        x=F.relu(x)
        # S2 MaxPool 2x2
        x = F.max_pool2d(x, 2)                
        # C3
        x = F.relu(self.conv2(x))
        # S4 
        x = F.max_pool2d(x, 2)
        # Flatten              
        x = self.flatten(x)
        # F5
        x = F.relu(self.fc1(x))        
        # F6 
        x = F.relu(self.fc2(x))
        # OUTPUT
        logits = self.fc3(x)

        probs = F.softmax(logits,dim=1) # y_pred: 0~1 
        return probs

```
{% endtab %}

{% tab title="myModel.py (2nd version)" %}


```python
# myModel.py

# ... same as Example 1


# Model Architecture - 2nd version using Sequential
class LeNet5v2(nn.Module):

    def __init__(self):
        super(LeNet5v2, self).__init__()
        
        # Feature Extraction 
        self.conv_layers = nn.Sequential(
            # C1
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            # S2
            nn.MaxPool2d(2, 2),
            # C3
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # S4
            nn.MaxPool2d(2, 2)
        )   
        
        self.flatten = nn.Flatten()
        
        # Classifier
        self.fc_layers = nn.Sequential(
            # F5
            nn.Linear(16*5*5,120),
            nn.ReLU(),
            # F6
            nn.Linear(120, 84),
            nn.ReLU(),
            # Output
            nn.Linear(84, 10)            
        )

    def forward(self, x):        
        # Feature Extraction
        x=self.conv_layers(x)        
        # Flatten              
        x = self.flatten(x)
        # Classification
        logits = self.fc_layers(x)
        
        probs = F.softmax(logits,dim=1) # y_pred: 0~1 
        return probs
```
{% endtab %}
{% endtabs %}



{% hint style="info" %}
The only changes in Example 2 are  (1) creating CNN model class  (2) changing the input image dimension. The rest of the code are the same
{% endhint %}

###

### Train and Evaluation of CNN Classification

> LeNet uses 1x32x32 input. Therefore, reshape MNIST from 1x28x28 to 1x32x32
>
> * MLP uses 1x28x28 as the Input for MNIST
> * LeNet uses 1x32x32 as the Input for MNIST



1.  Download the source code for **Model Training**: [TU\_PyTorch\_CNN\_Train.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)

    * Run this code and observe the loss
    * Change the training epoch


2. Download the source code for **Model Evaluation**: [TU\_PyTorch\_CNN\_Eval.py](https://github.com/ykkimhgu/DLIP-src/tree/main/Tutorial\_Pytorch/TU\_MLP\_CNN\_Classification\_ExampleCode)
   * Run this code and observe the evaluation accuracy and sample images





{% tabs %}
{% tab title="TU_PyTorch_CNN_Train.py" %}
````python
```python
##########################################################
# PyTorch Tutorial:  Classification CNN Model for Training
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: This example is creating and training a CNN model for classification
#
# Model: LeNet
# Dataset: MNIST
#
##########################################################


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt

# User defined Model
import myModel  as myModel 

# Model weight directory
ModelPATH='./weights/'

##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



##########################################################
## Part 1:  Prepare Dataset 
##########################################################

# Download Dataset from TorchVision MNIST
# Once, downloaded locally, it does not download again.
#
# NOTE: LeNet uses 1x32x32 input. 
# Reshape MNIST from 1x28x28 to  1x32x32

# transformation for resize
data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
])

# TRAIN DATA
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=data_transform,   #converts data size
)


# Create DataLoader with Batch size N
# Input Dim:  [N, C, H, W]
batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break




##########################################################
## Part 2:  Create Model Instance - CNN
##########################################################

# Model Class Construction
model = myModel.LeNet5().to(device)
print(model)



##########################################################
## Part 3:  Train Model
##########################################################

# Loss Function
loss_fn = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Run Train for k epoch
epochs = 1
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    myModel.train(train_dataloader, model,loss_fn, optimizer,device)    
print("Done!")


# Save Train Model
# * Need to create a new folder PATH priorly
torch.save(model,ModelPATH+'LeNet_MNIST_model.pth')



# Evaluate the trained model in **Eval.py
```
````
{% endtab %}

{% tab title="TU_PyTorch_CNN_Eval.py" %}
````python
```python
##########################################################
# PyTorch Tutorial:  Classification CNN Model for Evaluation
#
# Author: Y.K.Kim
# mod: 2024-5-21 
#
# Descr: This example is tesing pretrined CNN model for classification
#
# Model: LeNet
# Dataset: MNIST
#
##########################################################


import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 
import matplotlib.pyplot as plt

# User defined Model
import myModel as myModel

# Model weight directory
ModelPATH='./weights/'



##########################################################
## Part 0:  GPU setting
##########################################################

# Select GPU or CPU for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



##########################################################
## Part 1:  Prepare Dataset
##########################################################

# Download Dataset from TorchVision MNIST
# Once, downloaded locally, it does not download again.
#
# NOTE: LeNet uses 1x32x32 input. 
# Reshape MNIST from 1x28x28 to  1x32x32


# transformation for resize
data_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
])

# EVAL DATA
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=data_transform,
)


# Create DataLoader with Batch size N
# Input Dim:  [N, C, H, W]
batch_size = 64
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break



##########################################################
## Part 4:  Test Model - Evaluation
##########################################################

model=torch.load(ModelPATH+'LeNet_MNIST_model.pth')
myModel.test(test_dataloader, model, device)




##########################################################
## Part 5:  Visualize Evaluation Results
##########################################################

# Select one batch of images
dataiter = iter(test_dataloader)
images, labels = next(dataiter)
print(images.size())


# Evaluate on one batch test images
with torch.no_grad():
  for X, y in dataiter:
      X, y = X.to(device), y.to(device)
      
      # Prediction Label 
      pred = model(X)
      _, predicted = torch.max(pred.data, 1)


# Plot 9 output images
figure = plt.figure()
num_of_images = 9
for index in range(num_of_images):
    plt.subplot(3, 3, index+1)
    plt.axis('off')    
    plt.title("Predicted: {}".format(predicted[index].item()))
    plt.imshow(images[index].cpu().numpy().squeeze(), cmap='gray_r')
plt.show()


```
````
{% endtab %}
{% endtabs %}
