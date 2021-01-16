# autograd torch

## AUTOGRAD: AUTOMATIC DIFFERENTIATION

Central to all neural networks in PyTorch is the `autograd` package.

* provides automatic differentiation for all operations on Tensors.
* Generally speaking, `torch.autograd` is an engine for computing vector-Jacobian product.

  `torch.Tensor` is the central class of the package.

* `.requires_grad` as `True`, starts to track all operations on it
  * you can call `.backward()` and have all the gradients computed automatically.
  * accumulated into `.grad` attribute.
* `with torch.no_grad():` to prevent tracking history \(and using memory\)
* `.backward()`  compute the derivatives,
* `.grad_fn` attribute  references a `Function` 

{% embed url="https://github.com/ykkimhgu/gitbook\_docs/blob/master/deep-learning-framework/pytorch/autograd\_tutorial\_ykk.ipynb" caption="" %}

