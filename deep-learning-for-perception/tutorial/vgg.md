# VGG

## VGG16

**Reference**

* [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) \(ICLR 2015\)

By default, it loads weights pre-trained on ImageNet. Check 'weights' for other options.

## Keras

### Pretrained model:

from [Keras application of VGG 16, 19](https://keras.io/api/applications/vgg/#vgg16-function)

[My example colab code: click here](https://colab.research.google.com/drive/1yjiFt1BiTE7H8BduxJU-8hWSumH-KDW7#scrollTo=zHK9wFdofwor)

* Check the index of imagenet 1000 classes labels: [click here](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a)

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

Building from scratch



