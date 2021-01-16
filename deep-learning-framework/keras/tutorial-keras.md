# Tutorial Keras

Follow the tutorials in the following orders

### Keras.io Tutorial

#### Computer Vision Tutorials: [https://keras.io/examples/vision/](https://keras.io/examples/vision/)

1. **Simple MNIST CNN**
   * modified tutorial
     * [source from github](https://github.com/ykkimhgu/dl-tutorial/blob/master/Keras/cnn/keras_tutorial_MNIST_ykk.ipynb) ,  [run on colab](https://colab.research.google.com/github/ykkimhgu/dl-tutorial/blob/master/Keras/cnn/keras_tutorial_MNIST_ykk.ipynb)
   * original tutorial
     * [source  from github](https://github.com/keras-team/keras-io/blob/master/examples/vision/mnist_convnet.py) ,  [run on colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/mnist_convnet.ipynb)
2. **Image classification from scratch**
   * modified tutorial
     * [ source from github](https://github.com/ykkimhgu/dl-tutorial/blob/master/Keras/cnn/Keras_tutorial_image_classification_from_scratch.ipynb)
   * original tutorial
     * [source from github](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py) ,  [run on colab](https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/vision/ipynb/image_classification_from_scratch.ipynb#scrollTo=YSKPuW9AlAV8)

### Pyimagesearch.com Tutorial

#### Get Started Tutorial

1. [How to train and test **your own dataset**](https://www.pyimagesearch.com/2018/09/10/keras-tutorial-how-to-get-started-with-keras-deep-learning-and-python/) 

### Keras Model Zoo:

[Collection of pretrained models: click here](https://modelzoo.co/framework/keras)

* [YOLOv3 in Keras](https://modelzoo.co/model/keras-yolov3)
* [SSD in Keras](https://modelzoo.co/model/single-shot-multibox-detector-keras)

### Other Tutorials

{% embed url="https://medium.com/analytics-vidhya/a-start-to-finish-guide-to-building-deep-neural-networks-in-keras-3d54de097a75" caption="" %}

## Exercise

### Image Recognition

1. Use AlexNet on MNIST
2. Use LeNet to train and recognize vehicles. Use Dataset of 
3. Create your own convnet network for MNIST dataset. Compare the performance with tutorial output

## Popular network architecture in Keras

### Available models <a id="available-models"></a>

| Model | Size | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [Xception](https://keras.io/api/applications/xception) | 88 MB | 0.790 | 0.945 | 22,910,480 | 126 |
| [VGG16](https://keras.io/api/applications/vgg/#vgg16-function) | 528 MB | 0.713 | 0.901 | 138,357,544 | 23 |
| [VGG19](https://keras.io/api/applications/vgg/#vgg19-function) | 549 MB | 0.713 | 0.900 | 143,667,240 | 26 |
| [ResNet50](https://keras.io/api/applications/resnet/#resnet50-function) | 98 MB | 0.749 | 0.921 | 25,636,712 | - |
| [ResNet101](https://keras.io/api/applications/resnet/#resnet101-function) | 171 MB | 0.764 | 0.928 | 44,707,176 | - |
| [ResNet152](https://keras.io/api/applications/resnet/#resnet152-function) | 232 MB | 0.766 | 0.931 | 60,419,944 | - |
| [ResNet50V2](https://keras.io/api/applications/resnet/#resnet50v2-function) | 98 MB | 0.760 | 0.930 | 25,613,800 | - |
| [ResNet101V2](https://keras.io/api/applications/resnet/#resnet101v2-function) | 171 MB | 0.772 | 0.938 | 44,675,560 | - |
| [ResNet152V2](https://keras.io/api/applications/resnet/#resnet152v2-function) | 232 MB | 0.780 | 0.942 | 60,380,648 | - |
| [InceptionV3](https://keras.io/api/applications/inceptionv3) | 92 MB | 0.779 | 0.937 | 23,851,784 | 159 |
| [InceptionResNetV2](https://keras.io/api/applications/inceptionresnetv2) | 215 MB | 0.803 | 0.953 | 55,873,736 | 572 |
| [MobileNet](https://keras.io/api/applications/mobilenet) | 16 MB | 0.704 | 0.895 | 4,253,864 | 88 |
| [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function) | 14 MB | 0.713 | 0.901 | 3,538,984 | 88 |
| [DenseNet121](https://keras.io/api/applications/densenet/#densenet121-function) | 33 MB | 0.750 | 0.923 | 8,062,504 | 121 |
| [DenseNet169](https://keras.io/api/applications/densenet/#densenet169-function) | 57 MB | 0.762 | 0.932 | 14,307,880 | 169 |
| [DenseNet201](https://keras.io/api/applications/densenet/#densenet201-function) | 80 MB | 0.773 | 0.936 | 20,242,984 | 201 |
| [NASNetMobile](https://keras.io/api/applications/nasnet/#nasnetmobile-function) | 23 MB | 0.744 | 0.919 | 5,326,716 | - |
| [NASNetLarge](https://keras.io/api/applications/nasnet/#nasnetlarge-function) | 343 MB | 0.825 | 0.960 | 88,949,818 | - |
| [EfficientNetB0](https://keras.io/api/applications/efficientnet/#efficientnetb0-function) | 29 MB | - | - | 5,330,571 | - |
| [EfficientNetB1](https://keras.io/api/applications/efficientnet/#efficientnetb1-function) | 31 MB | - | - | 7,856,239 | - |
| [EfficientNetB2](https://keras.io/api/applications/efficientnet/#efficientnetb2-function) | 36 MB | - | - | 9,177,569 | - |
| [EfficientNetB3](https://keras.io/api/applications/efficientnet/#efficientnetb3-function) | 48 MB | - | - | 12,320,535 | - |
| [EfficientNetB4](https://keras.io/api/applications/efficientnet/#efficientnetb4-function) | 75 MB | - | - | 19,466,823 | - |
| [EfficientNetB5](https://keras.io/api/applications/efficientnet/#efficientnetb5-function) | 118 MB | - | - | 30,562,527 | - |
| [EfficientNetB6](https://keras.io/api/applications/efficientnet/#efficientnetb6-function) | 166 MB | - | - | 43,265,143 | - |
| [EfficientNetB7](https://keras.io/api/applications/efficientnet/#efficientnetb7-function) | 256 MB | - | - | 66,658,687 | - |

The top-1 and top-5 accuracy refers to the model's performance on the ImageNet validation dataset.

Depth refers to the topological depth of the network. This includes activation layers, batch normalization layers etc.

### Inception

### ResNet

### VGG

### Squeeze Net

### YOLOv3

