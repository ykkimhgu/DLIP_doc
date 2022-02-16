# MNIST using LeNet

## MNIST Convnet Tutorial

**Modified: Y.-K Kim 2020-08-19**

\*\*\*\*[**Source code**](https://github.com/ykkimhgu/gitbook\_docs/blob/master/deep-learning-framework/pytorch/MNIST\_tutorial\_ykk.ipynb)\*\*\*\*

It is a simple feed-forward CNN network for MNIST. Lets use LeNet for MNIST handwritten recognition.

A typical training procedure for a neural network is as follows:

* Prepare dataset of inputs
* Define the neural network that has some learnable parameters&#x20;
* Compute the loss (how far is the output from being correct)
* Propagate gradients back into the network’s parameters
*   Update the weights of the network, typically using a simple update rule:

    `weight = weight - learning_rate * gradient`

#### MNIST Dataset

It is a collection of 70000 handwritten digits split into training and test set of 60000 and 10000 images respectively.

> Note: expected input size of this net (LeNet) is 32x32. To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.

```python
%matplotlib inline

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#  a batch_size of 64, size 1000 for testing
#  mean 0.1307, std 0.3081 used for the Normalize() 
batch_size_train=64
batch_size_test=64

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])   
transform = transforms.Compose(
    [transforms.Resize((32, 32)),transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])  

# Train set    
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True, num_workers=2)

# Test set
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=True, num_workers=2)
```

Let us check out the shape of the images and the labels.

```python
print(len(trainloader.dataset))
print(len(testloader.dataset))

dataiter = iter(trainloader)
images, labels = dataiter.next()

print(images.shape)
print(labels.shape)
```

```
60000
10000
torch.Size([64, 1, 32, 32])
torch.Size([64])
```

#### Plot some train data

```python
import matplotlib.pyplot as plt

figure = plt.figure()
num_of_images = 9
for index in range(num_of_images):
    plt.subplot(3, 3, index+1)
    plt.axis('off')    
    plt.title("Ground Truth: {}".format(labels[index]))
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
```

![](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F0sVPN2R8JjyYw6QCaSkv%2Ffile.png?alt=media)

### Define the network

Let’s define this network:

> tensor.view(-1,n), Returns a new tensor with the same data as the self tensor but of a different shape. the size -1 is inferred from other dimensions

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input ch, output ch, convolution
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = self.pool(x)        
        x = F.relu(self.conv2(x))
        x = self.pool(x)        
        #x = x.view(-1, self.num_flat_features(x))        
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#net = Net()
#print(net)
```

You just have to define the `forward` function, and the `backward` function (where gradients are computed) is automatically defined for you using `autograd`. You can use any of the Tensor operations in the `forward` function.

### Loss Function and Optimization

A loss function takes the (output, target) pair of inputs, and computes a value that estimates how far away the output is from the target. Define loss function as loss=criterion(outputs, labels)

```python
import torch.optim as optim

# loss function
criterion = nn.CrossEntropyLoss()
# Optimization method
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Zero the gradient buffers of all parameters and backprops with random gradients:

### Train network

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]        
        images, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished Training')
```

```
[1,   100] loss: 0.291
[1,   200] loss: 0.260
[1,   300] loss: 0.283
[1,   400] loss: 0.316
[1,   500] loss: 0.283
[1,   600] loss: 0.324
[1,   700] loss: 0.280
[1,   800] loss: 0.257
[1,   900] loss: 0.286
[2,   100] loss: 0.224
[2,   200] loss: 0.243
[2,   300] loss: 0.279
[2,   400] loss: 0.235
[2,   500] loss: 0.235
[2,   600] loss: 0.281
[2,   700] loss: 0.303
[2,   800] loss: 0.340
[2,   900] loss: 0.270
[3,   100] loss: 0.236
[3,   200] loss: 0.213
[3,   300] loss: 0.259
[3,   400] loss: 0.296
[3,   500] loss: 0.227
[3,   600] loss: 0.288
[3,   700] loss: 0.213
[3,   800] loss: 0.235
[3,   900] loss: 0.325
[4,   100] loss: 0.235
[4,   200] loss: 0.233
[4,   300] loss: 0.196
[4,   400] loss: 0.226
[4,   500] loss: 0.242
[4,   600] loss: 0.238
[4,   700] loss: 0.222
[4,   800] loss: 0.267
[4,   900] loss: 0.258
[5,   100] loss: 0.222
[5,   200] loss: 0.258
[5,   300] loss: 0.203
[5,   400] loss: 0.207
[5,   500] loss: 0.239
[5,   600] loss: 0.257
[5,   700] loss: 0.164
[5,   800] loss: 0.210
[5,   900] loss: 0.282
Finished Training
```

### Save Model

```python
PATH = './MNIST_net.pth'
torch.save(net.state_dict(), PATH)
```

### Test the network on the test data

We have trained the network for 2 passes over the training dataset. But we need to check if the network has learnt anything at all.

We will check this by predicting the class label that the neural network outputs, and checking it against the ground-truth. If the prediction is correct, we add the sample to the list of correct predictions.

Okay, first step. Let us display an image from the test set to get familiar.

```python
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:        
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the %d test images: %d %%' %(len(testloader.dataset), 100 * correct / total))
```

```
Accuracy of the network on the 10000 test images: 98 %
```

### Visualize test results

You need to covert from GPU to Tensor.cpu() . e.g. images.cpu()

```python
figure = plt.figure()
num_of_images = 9
for index in range(num_of_images):
    plt.subplot(3, 3, index+1)
    plt.axis('off')    
    plt.title("Predicted: {}".format(predicted[index].item()))
    plt.imshow(images[index].cpu().numpy().squeeze(), cmap='gray_r')
```

![](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FK8dDCkdNzdVPgBhchAbN%2Ffile.png?alt=media)

### Continued Training from Checkpoints

see how we can continue training from the state\_dicts we saved during our first training run.

```python
continued_net=Net().to(device)
continued_net.load_state_dict(torch.load(PATH))
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
continued_net.to(device)

def train_continue(epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]        
        images, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = continued_net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0



for epoch in range(6,10):  # loop over the dataset multiple times
    train_continue(epoch)

print('Finished Training')
```

```
[7,   100] loss: 0.160
[7,   200] loss: 0.209
[7,   300] loss: 0.222
[7,   400] loss: 0.243
[7,   500] loss: 0.186
[7,   600] loss: 0.240
[7,   700] loss: 0.158
[7,   800] loss: 0.187
[7,   900] loss: 0.167
[8,   100] loss: 0.194
[8,   200] loss: 0.189
[8,   300] loss: 0.167
[8,   400] loss: 0.189
[8,   500] loss: 0.231
[8,   600] loss: 0.170
[8,   700] loss: 0.173
[8,   800] loss: 0.194
[8,   900] loss: 0.202
[9,   100] loss: 0.189
[9,   200] loss: 0.223
[9,   300] loss: 0.202
[9,   400] loss: 0.235
[9,   500] loss: 0.196
[9,   600] loss: 0.167
[9,   700] loss: 0.194
[9,   800] loss: 0.206
[9,   900] loss: 0.146
[10,   100] loss: 0.172
[10,   200] loss: 0.230
[10,   300] loss: 0.185
[10,   400] loss: 0.226
[10,   500] loss: 0.174
[10,   600] loss: 0.208
[10,   700] loss: 0.184
[10,   800] loss: 0.175
[10,   900] loss: 0.219
Finished Training
```

```python
# Accuracy of continued training

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:        
        images, labels = data[0].to(device), data[1].to(device)
        outputs = continued_net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


print('Accuracy of the network on the %d test images: %d %%' %(len(testloader.dataset), 100 * correct / total))
```

```
Accuracy of the network on the 10000 test images: 98 %
```

```python
```
