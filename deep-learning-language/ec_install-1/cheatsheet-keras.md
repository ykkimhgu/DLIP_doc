# Cheat Sheet

## Keras API cheat sheet



### Check GPU

```text
device_name = tf.test.gpu_device_name()
print('GPU at: {}'.format(device_name))
```

### Check Library Version

```text
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
print(keras.__version__)

import numpy as np
print(np.__version__)
```

### Split into train validate database

```text
#0.8 for train, 0.2 for validation(testing)
# data_path is database folder path

image_size = (180, 180)
batch_size = 16
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=data_path,
    subset="training",
    validation_split=0.2,     
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

### Preprocessing Database

#### Buffer Prefetch

```text
# prefetch data to GPU
train_ds = train_ds.prefetch(buffer_size=batch_size)
val_ds = val_ds.prefetch(buffer_size=batch_size)
```

#### Rescaling, Cropping - included in model

```text
# x is from  inputs = keras.Input(shape=input_shape)
# Rescale 0 to 1
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
# Center-crop images to 180x180
x = layers.experimental.preprocessing.CenterCrop(height=180, width=180)(x)
```

#### 

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
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
```

### Save and load model in Keras <a id="How-to-save-and-load-Model-in-Keras"></a>

#### Model and Weight in one file

```text
# save model and architecture to single file
model.save("model.h5")

# load model
from keras.models import load_model
model = load_model('model.h5')

```

#### Model \(json\) and weight separately

```text
# serialize model to JSON
from keras.models import model_from_json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
```

### Run inference 

#### Test on some data

```text
img = keras.preprocessing.image.load_img(
    "PetImages/Cat/6779.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(img_array)
score = predictions[0]
print(
    "This image is %.2f percent cat and %.2f percent dog."
    % (100 * (1 - score), 100 * score)
)
```

#### Test on all validate database

