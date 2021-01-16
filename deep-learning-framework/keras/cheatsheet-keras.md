# Cheat Sheet

&gt; ! create an extensive notebook for this cheat sheet

## Keras API cheat sheet

### Check Library Version

```text
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import layers
print(keras.__version__)

import numpy as np
print(np.__version__)

import cv2
print(cv2.__version__)
```

### Check GPU

```text
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

device_name = tf.test.gpu_device_name()
print('GPU at: {}'.format(device_name))
```

### Prepare Datasets

#### Option1\) Use datasets provided by TF/Keras

The tf.keras.datasets module provide a few toy datasets \(already-vectorized, in Numpy format\) that can be used for debugging a model or creating simple code examples. If you are looking for larger & more useful ready-to-use datasets, take a look at TensorFlow Datasets.

> TF datasets have different format and functions

Keras Dataset laod functions return Tuple of Numpy arrays: \(x\_train, y\_train\), \(x\_test, y\_test\).

* MNIST digits classification dataset
* CIFAR10 small images classification dataset
* CIFAR100 small images classification dataset
* Fashion MNIST dataset, an alternative to MNIST etc..

It downloads and saves dataset in local drive \(~/.keras/datasets\)

```python
# MNIST
# This is a dataset of 60,000 28x28 grayscale images of the 10 digits, along with a test set of 10,000 images.
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")

# CIFAR10
# This is a dataset of 50,000 32x32 color training images and 10,000 test images, labeled over 10 categories.
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar10.load_data()
```

#### Option2\) Use or create your own database in local storage <a id="Option2)-Use-or-create-your-own-database-in-local-storage"></a>

Example: MS Cats vs Dogs images dataset

```text
# Download the 786M ZIP archive of the raw data from 

!curl -O https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
!unzip -q kagglecatsanddogs_3367a.zip
!ls

# Now we have a PetImages folder which contain two subfolders, Cat and Dog. Each subfolder contains image files for each category.
```

* Assume raw data is downloaded and `PetImages` folder with two subfolders, `Cat` and `Dog` is saved locally.

  Example: ~.keras/datasets/PetImages/

```python
import os

# Check the current working directory
display(os. getcwd())

# add 'r' before the directory path  for unicode error
folder_base_path=r"C:/Users/ykkim/.keras/datasets/PetImages/"
dir_list = os.listdir(folder_base_path) 
print(dir_list)
```

* Filter out corrupted images

  When working with lots of real-world image data, corrupted images are a common occurence. Let's filter out badly-encoded images that do not feature the string "JFIF" in their header.

```text
num_skipped = 0
for folder_name in ("Cat", "Dog"):
    folder_path = os.path.join(folder_base_path, folder_name)
    #print(os.listdir(folder_path))
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        #display(fpath)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()
        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)
```

### Load and Plot Images

#### Using OpenCV  \(color mode is  B-G-R\)

```python
import cv2
from matplotlib import pyplot as plt


# Load and plot a sample image
img_file= folder_base_path+ "Cat/1.jpg"
img = cv2.imread(img_file) 
cv2.imshow('image', img) 
print(type(img))
print(img.shape)
print(img.dtype)

# MAT also can be plotted using pyplot
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()
```

#### Using Matplotlib

```text
# load and display an image with Matplotlib

from matplotlib import image
from matplotlib import pyplot

# load image as pixel array
img = image.imread(img_file)

# summarize shape of the pixel array
print(img.dtype)
print(img.shape)

# display the array of pixels as an image
pyplot.imshow(img)
pyplot.show()
```

#### Load and plot using PIL <a id="Load-and-plot-using-PIL"></a>

```python
#Load and Plot image in Keras - PIL image
img = keras.preprocessing.image.load_img(
    img_file, grayscale=False
)

print(img.format)
print(img.mode)
print(img.size)

# show the image in new window
img.show()
plt.imshow(img) # can also use plt
```

#### Convert PIL to Numpy,  OpenCV to Numpy

```text
#Convert PIL image into Numpy Array  
img_array = keras.preprocessing.image.img_to_array(img)
print(img_array.shape)  # (32=batch,180,180, channel=3)
print(img_array.dtype)  # float32 
plt.imshow(img_array.astype("uint8"))

#Convert MAT into Numpy Array
```

#### Subplot with matplotlib

```text
from matplotlib import image

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)        
    img_file= folder_base_path+"Cat/"+str(i)+".jpg"        
    images = image.imread(img_file)    
    plt.imshow(images.astype("uint8"))
    plt.axis("off")
```

### Split into train validate database

#### Option 1\) Classes divided by folder name. `image_dataset_from_directory` <a id="Option-1)--Classes-divided-by-folder-name.--image_dataset_from_directory"></a>

No Train/valid/Test folders

Generates a 'tf.data.Dataset' from image files in a directory.

If your directory structure is:

```text
main_directory/
...class_a/
......a_image_1.jpg
......a_image_2.jpg
...class_b/
......b_image_1.jpg
......b_image_2.jpg
```

return a 'tf.data.Dataset' that yields batches of images _class\_a with label=0_, _class\_b with label=1_

```text
#This works only for  TF-nightly, higher than TF2.2.0

#tf.keras.preprocessing.image_dataset_from_directory(
#    directory, labels='inferred', label_mode='int', class_names=None,
#    color_mode='rgb', batch_size=32, image_size=(256, 256), shuffle=True, seed=None,
#    validation_split=None, subset=None, interpolation='bilinear', follow_links=False)


# folder_base_path=r"C:/Users/ykkim/.keras/datasets/PetImages/"
data_path = folder_base_path
display(data_path)

image_size = (180, 180)
batch_size = 16

# 0.8 for train, 0.2 for validation.  Valid==testing here 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_path,
    validation_split=0.2,     
    subset="training",     #"training" or "validation"
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_path,
    validation_split=0.2,        
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
```

#### Option 2\) Train Valid Test are divided by folder names manually `flow_from_directory` <a id="Option-2)-Train-Valid-Test-are-divided-by-folder-names-manually--flow_from_directory"></a>

![The directory structure for a binary classification problem](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/images/image%20%281%29.png)

```text
# Assuming datasets are divided in  '/train',''/valid',''/test' folders

train_generator = train_datagen.flow_from_directory(
    directory=r"./train/",
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",  #"categorical", "binary", "sparse", "input", or None
    shuffle=True,
    seed=42
)

valid_generator = valid_datagen.flow_from_directory(
    directory=r"./valid/",
    target_size=image_size,
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=r"./test/",
    target_size=image_size,
    color_mode="rgb",
    batch_size=1,
    class_mode=None,
    shuffle=False,
    seed=42
)
```

### Visualize the dataset

```text
import matplotlib.pyplot as plt

#Creates a Dataset with batch=1
for images, labels in train_ds.take(1):   # taking one batch
        print(images.shape)  # (32=batch,180,180, channel=3)
        print(images.dtype)  # float32
        print(labels.shape)  # (32=batch
        print(labels.dtype)  # int32    
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
```

### Preprocessing Database

#### Buffer Prefetch

```text
# prefetch data to GPU
train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)
```

#### Rescaling, Cropping - can be included in model

```text
# x is from  inputs = keras.Input(shape=input_shape)
# Rescale 0 to 1
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
# Center-crop images to 180x180
x = layers.experimental.preprocessing.CenterCrop(height=180, width=180)(x)
```

### Build Model

* **Example 1: A few layer CNN for a simple exampl**e

```text
input_shape=image_size + (3,)
num_classes=2;

print(input_shape)

model=keras.Sequential(
    [
     keras.Input(shape=input_shape),
     layers.Conv2D(32,kernel_size=(3,3), activation="relu"), #(filerNumber=32)
     layers.MaxPooling2D(pool_size=(2,2)),
     layers.Conv2D(64,kernel_size=(3,3), activation="relu"),
     layers.Flatten(),
     layers.Dropout(0.5),
     layers.Dense(num_classes,activation="softmax"),
    ]
)
model.summary()
```

#### \* Example 2: Small version of Xception

```text
#  Small version of the Xception network. 

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)


    # PREPROCESSING for Model Input
    # Image augmentation block with flip, rotation
    x = data_augmentation(inputs)
    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # Center-crop images to 180x180
    #x = layers.experimental.preprocessing.CenterCrop(height=180, width=180)(x)

    # Building Model
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)  # filter=32, kernelSize=3x3
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,), num_classes=2)
```

#### For other archiectures, go to Tutorial

### Visualize model

```text
# Need pydot, graphviz
# for Conda:   conda install -c anaconda pydot
keras.utils.plot_model(model, show_shapes=True)

# summarize model in text
model.summary()
```

### Train the model

```text
# Example 1
epochs = 2

my_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2),
    tf.keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_ds, epochs=epochs, callbacks=my_callbacks, validation_data=val_ds,
)
```

### Save and load model in Keras <a id="How-to-save-and-load-Model-in-Keras"></a>

#### Option 1\) Model and Weight in one file \(_gives error..._ \)

```text
# save model and architecture to single file
model.save("model.h5")

# load model
from keras.models import load_model
model = load_model('model.h5')
```

#### Option 2\) Model \(json\) and weight separately

```text
from keras.models import model_from_json

# SAVE model and weight
# serialize model to JSON
model_json = model.to_json()
with open("model_xception_catdog.json", "w") as json_file:
    json_file.write(model_json)    
# serialize weights to HDF5
model.save_weights("weight_xception_catdog.h5")
print("Saved model to disk")


# LOAD model and weight
with open('model_xception_catdog.json','r') as f:    
    loaded_model = tf.keras.models.model_from_json(f.read())
loaded_model.load_weights("weight_xception_catdog.h5")
```

### Run inference

#### Test on some data

```text
# Example 1
display(data_path+"/Cat/6779.jpg")

#Load and Plot image in Keras - PIL image
img = keras.preprocessing.image.load_img(
    data_path+"Cat/6779.jpg", target_size=image_size, grayscale=False
)
display(img.size)
img.show()

#Convert PIL image into Numpy Array  
img_array = keras.preprocessing.image.img_to_array(img)
plt.imshow(img_array.astype("uint8"))

# Reshape NumpyArray to  Model input size (batchx180x180x3)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis
display(img_array.shape)

predictions = model.predict(img_array)
display(predictions.shape)
score = predictions[0][0]

#print format: e.g.  print("pi=%s" % "3.14159")
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)
```

#### Test on all validate database

```text
# Evaluate the whole validation dataset
score=model.evaluate(val_ds, verbose=0)
print(score.shape)
print("Test loss:", score[0])
print("Test accuracy: ", score[1])
```

