# Cheat Sheet

## Numpy &lt;-&gt; Torch Tensor

Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.

#### Converting a Torch Tensor to a NumPy Array

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
```

#### Converting NumPy Array to Torch Tensor

See how changing the np array changed the Torch Tensor automatically

```text
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
```

## Install tensorflow

## Install Keras

