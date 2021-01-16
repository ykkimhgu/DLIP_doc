# LeNet-5 Tutorial

## Introduction

Overview of LeNet: click here

* Activation Function: TanH
* Pooling: Avg. pooling
* No Padding
* F.C: softmax
  * Originally used RBF\(Radial Basis Function\)
* Loss Function: MSE 
* Input: 32x32x1
  * MNIST image is 28x28.  MNIST is padded to 32

![](../../../images/image%20%28231%29.png)

### LeNet-5 layers: <a id="d723"></a>

1. Convolution \#1. Input = 32x32x1. Output = 28x28x6 `conv2d`
2. SubSampling \#1. Input = 28x28x6. Output = 14x14x6. SubSampling is simply Average Pooling so we use `avg_pool`
3. Convolution \#2. Input = 14x14x6. Output = 10x10x16 `conv2d`
4. SubSampling \#2. Input = 10x10x16. Output = 5x5x16 `avg_pool`
5. Fully Connected \#1. Input = 5x5x16. Output = 120
6. Fully Connected \#2. Input = 120. Output = 84
7. Output 10

## Keras

[Another code example: click here](https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/)

[Full code: click here](https://colab.research.google.com/drive/18FSrS80KtvRW5-bedEQ3HwDKelbNfUSy#scrollTo=5zp3oRg6lP0d)

```python
model = keras.Sequential()

model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(32,32,1)))
model.add(layers.AveragePooling2D())

model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
model.add(layers.AveragePooling2D())

model.add(layers.Flatten())

model.add(layers.Dense(units=120, activation='relu'))

model.add(layers.Dense(units=84, activation='relu'))

model.add(layers.Dense(units=10, activation = 'softmax'))
```

## PyTorch

[Sample code: click here](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

> Originally CONV 5x5. Some code use CONV 3x3

```python
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

```python
output = net(input)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
```

