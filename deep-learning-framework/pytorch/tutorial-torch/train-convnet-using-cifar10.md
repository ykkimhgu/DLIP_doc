# Train ConvNet using CIFAR10

### Training an image classifier <a id="Training-an-image-classifier"></a>

We will do the following steps in order:

1. Load and normalizing the CIFAR10 training and test datasets using `torchvision`
2. Define a Convolutional Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
6. Loading and normalizing CIFAR10 

### Data Loading

```text
import torch
import torchvision
import torchvision.transforms as transforms
```

> The output of torchvision datasets are PILImage images of range \[0, 1\]. We transform them to Tensors of normalized range \[-1, 1\].

#### Download General Dataset

{% embed url="https://pytorch.org/docs/stable/torchvision/datasets.html" %}

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

#### Use Downloaded General Dataset

Change the option `download=False`,  and set the  path \(`root)`where data is stored.

```python
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
                                        
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
```

#### Use User Defined Dataset

