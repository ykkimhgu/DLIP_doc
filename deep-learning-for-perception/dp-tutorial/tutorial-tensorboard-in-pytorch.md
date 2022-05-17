# Tutorial: Tensorboard in Pytorch



Follow Tutorial here



{% embed url="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" %}



## TensorBoard in PyTorch

In this tutorial, we implement a MNIST classifier using a simple neural network and visualize the training process using [TensorBoard](https://www.tensorflow.org/get\_started/summaries\_and\_tensorboard). In training phase, we plot the loss and accuracy functions through `scalar_summary` and visualize the training images through `image_summary`. In addition, we visualize the weight and gradient values of the parameters of the neural network using `histogram_summary`. PyTorch code for handling these summary functions can be found [here](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/main.py#L81-L97).

![](https://github.com/yunjey/pytorch-tutorial/raw/master/tutorials/04-utils/tensorboard/gif/tensorboard.gif)

\


### Usage

**1. Install the dependencies**

```
$ pip install -r requirements.txt
```

**2. Train the model**

```
$ python main.py
```

**3. Open the TensorBoard**

To run the TensorBoard, open a new terminal and run the command below. Then, open [http://localhost:6006/](http://localhost:6006/) on your web browser.

```
$ tensorboard --logdir='./logs' --port=6006
```

