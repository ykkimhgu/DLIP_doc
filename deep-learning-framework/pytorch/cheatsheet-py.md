# Cheat Sheet

## Numpy &lt;-&gt; Torch Tensor

Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

### Converting a Torch Tensor to a NumPy Array

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
```

### Converting NumPy Array to Torch Tensor

See how changing the np array changed the Torch Tensor automatically

```text
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

## Datasets

There are several options for dataset

* load data into a numpy array, then convert this array into a `torch.*Tensor`
* For images, packages such as Pillow, OpenCV are useful
* \(Recommend\) For vision, we have created a package called `torchvision`
  * data loaders for common datasets such as Imagenet, CIFAR10, MNIST, etc.

## Using GPU

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

for epoch in range(epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

# ...
# evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
# ...
```

## Visualizing Loss Curve

## Evaluation

## Model function

make it a function

## Train function

## Test function

## Plot output image function

