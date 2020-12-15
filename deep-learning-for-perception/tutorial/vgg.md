# VGG Tutorial

## Introduction

[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) \(ICLR 2015\)

Read about VGG: click here

### VGG-16

![](../../.gitbook/assets/image%20%28243%29.png)

![](../../.gitbook/assets/image%20%28238%29.png)

### VGG-19

![](../../.gitbook/assets/image%20%28235%29.png)

## Keras

### Pretrained model

Using [Keras application of VGG 16, 19](https://keras.io/api/applications/vgg/#vgg16-function) with ImageNet pretrained

* Check the index of imagenet 1000 classes labels: [click here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

[My example colab code: click here](https://colab.research.google.com/drive/1yjiFt1BiTE7H8BduxJU-8hWSumH-KDW7#scrollTo=zHK9wFdofwor)

```python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

# Open VGG model
model = VGG16(weights='imagenet')

img_path = 'cat2.jpg'
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Run Classfication
preds = model.predict(x)

# Display the score
display(preds.shape)
idx=np.argmax(preds)
score=preds[0][idx]
display(idx, score)

```

### Building from scratch

[VGG-16: My Keras code](https://colab.research.google.com/drive/1TUI3WX639yajO0Hf6KW-GsQ8VCZxPFod?usp=sharing),  [VGG-16 weight file](https://drive.google.com/u/1/uc?id=0Bz7KyqmuGsilT0J5dmRCM0ROVHc&export=download)

[Read this blog for step by step tutorial](https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c)



![](../../.gitbook/assets/image%20%28252%29.png)

```python
#Importing library
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np

np.random.seed(1000)


model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=2, activation="softmax"))

#Model Summary
model.summary()

#weights_path='vgg16_weights.h5'
#model.load_weights(weights_path)
```

## PyTorch

### Pretrained model:



### Building from scratch:

Implementation by PyTorch: [Vgg.py](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)







