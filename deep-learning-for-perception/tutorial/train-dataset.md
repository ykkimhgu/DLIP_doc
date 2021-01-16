# Train Dataset

Learn how to train a model with a train dataset in Keras

Source file needed

* Dataset
* model.py, trainmodel.py 

## Preparation

### Dataset

#### Download directly from Keras

### CNN model

> See other tutorials of how to build a model

Template code for my[model.py](https://colab.research.google.com/drive/1rEEn-FmGyi29p9q-OMxV8N8MD-u5T2Tq#scrollTo=0kX-r2kKksrE)

```python
#Importing library
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Define your model
def MYMODEL(weights_path=None):
    model = Sequential()

    # architecture goes here
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

    if weights_path:
    model.load_weights(weights_path)

    return model
```

## Train model

Read dataset

Preprocessing for correct input size

Train with an optimizer

```python
from keras.optimizers import Adam
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
```

Save model and weight file

```python
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
```

Show train results on validation set

```python
import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
```

## Further resource

[Tutorial: How to retune from pretrained model \(transfer learning\): VGG](https://www.youtube.com/watch?v=H8sXcAXrGR4&feature=youtu.be)

* [https://github.com/Hvass-Labs/TensorFlow-Tutorials](https://github.com/Hvass-Labs/TensorFlow-Tutorials)
* 
