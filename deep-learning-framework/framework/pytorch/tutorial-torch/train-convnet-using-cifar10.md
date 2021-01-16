# Train ConvNet using CIFAR10

[Source code \(Jupyter notebook\)](https://github.com/ykkimhgu/gitbook_docs/blob/master/deep-learning-framework/pytorch/neural_networks_tutorial_ykk.ipynb)

## Training an image classifier <a id="Training-an-image-classifier"></a>

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using `torchvision`
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

## Data Loading

```text
import torch
import torchvision
import torchvision.transforms as transforms
```

> The output of torchvision datasets are PILImage images of range \[0, 1\]. We transform them to Tensors of normalized range \[-1, 1\].

### Download General Dataset

{% embed url="https://pytorch.org/docs/stable/torchvision/datasets.html" caption="" %}

```python
# Transform: normalize a tensor image wih mead/std
# torchvision.transforms.Normalize(mean, std, inplace=False)
# For 3 channel image
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# torchvision.datasets.CIFAR10(root, train=True, transform=None, target_transform=None, download=False)
# Train set
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

# Test set
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

### Use Downloaded General Dataset

Change the option `download=False`, and set the path \(`root)`where data is stored.

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
```

### User Defined Dataset

## Load and Show images\(tensor, color\)

```python
import matplotlib.pyplot as plt
import numpy as np

# Cannot directly use plt.show() to show Tensor
# Convert to numpy then use plt
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # channel goes at last
    plt.xticks([])
    plt.yticks([])
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()


# Since batch=4, we get four images at a time
images.size()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

## Define Model

```python
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
```

## Define Loss function and Optimization

```python
import torch.optim as optim

# loss function
criterion = nn.CrossEntropyLoss()
# Optimization method
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

## Train the network

```python
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

## Save model

```python
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# To use the saved model
net = Net()
net.load_state_dict(torch.load(PATH))
```

## Test the network

### Show some ground truth of test data

```python
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
```

### Overall accuracy

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

### Evaluate each class

> numpy.squeeze\(\) function is used when we want to remove single-dimensional entries from the shape of an array.

```python
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)  # max(outputs, dim=1) returns (values, indices)
        c = (predicted == labels).squeeze()   # remove dim=1
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item() #c[i] is a tensor either true or false
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
```

## Exercise

* Try increasing the width of your network \(argument 2 of

  the first `nn.Conv2d`, and argument 1 of the second `nn.Conv2d`

  they need to be the same number\), see what kind of speedup you get.

* Build a MNIST Convnet

