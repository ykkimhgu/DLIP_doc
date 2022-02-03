# Eye Blinking Detection Alarm

**Date: 2021-6-21**

**Author**: Yoojung Yoon(21700483), Kungmin Jo(21701052)

**Github:** [https://github.com/Jo951128/eye\_blink\_detector](https://github.com/Jo951128/eye\_blink\_detector)

## Introduction

This tutorial is about detecting blinks. This is to prevent drowsy driving, which is consistently mentioned as a cause of traffic accidents. Drowsy driving is the highest cause of death on highways where serious accidents can occur. Therefore, drowsy driving is a common cause of accidents, and countermeasures for the problem are required. The tutorial is based on colab and Visual Code ver. It consists of two. In colab, it was cumbersome to load a webcam or video, so I replaced it with a captured image.

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FcCo86Ffvrvl1WcQ2kTUh%2Ffile.png?alt=media)

## Colab version

### Download files using git clone

* eye\_blink\_detector
  * dataset
  * model
  * eye\_blink.py
  * train.py
  * shape\_predictor\_68\_face\_landmarks.dat
  * requirements.txt

```python
%cd /content
!git clone https://github.com/Jo951128/eye_blink_detector.git
```

```
/content
Cloning into 'eye_blink_detector'...
remote: Enumerating objects: 34, done.[K
remote: Counting objects: 100% (34/34), done.[K
remote: Compressing objects: 100% (28/28), done.[K
```

### Traing

#### library load

```python
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Activation, Conv2D, Flatten, Dense, MaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
plt.style.use('dark_background')
```

#### Load Dataset

```python
x_train = np.load('eye_blink_detector/dataset/x_train.npy').astype(np.float32)
y_train = np.load('eye_blink_detector/dataset/y_train.npy').astype(np.float32)
x_val = np.load('eye_blink_detector/dataset/x_val.npy').astype(np.float32)
y_val = np.load('eye_blink_detector/dataset/y_val.npy').astype(np.float32)

print("train size : ", x_train.shape)
print("validation size : ", x_val.shape)
```

#### Preview

```python
import random
plt.subplot(2, 1, 1) 
random_train_num = random.randrange(1,2585) 
random_val_num = random.randrange(1,287)

plt.title(str(y_train[random_train_num])) #train set 
plt.imshow(x_train[random_train_num].reshape((26, 34)), cmap='gray')

plt.subplot(2, 1, 2)
plt.title(str(y_val[random_val_num])) #validataion set 
plt.imshow(x_val[random_val_num].reshape((26, 34)), cmap='gray')
```

#### Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    x=x_train, y=y_train,
    batch_size=32,
    shuffle=True
)

val_generator = val_datagen.flow(
    x=x_val, y=y_val,
    batch_size=32,
    shuffle=False
)
```

#### Build Model

```python
inputs = Input(shape=(26, 34, 1))

net = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(net)
net = MaxPooling2D(pool_size=2)(net)

net = Flatten()(net)

net = Dense(512)(net)
net = Activation('relu')(net)
net = Dense(1)(net)
outputs = Activation('sigmoid')(net)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.summary()
```

#### Model train

```python
start_time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

model.fit_generator(
    train_generator, epochs=50, validation_data=val_generator,
    callbacks=[
        ModelCheckpoint('models/%s.h5' % (start_time), monitor='val_acc', save_best_only=True, mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_acc', factor=0.2, patience=10, verbose=1, mode='auto', min_lr=1e-05)
    ]
)
```

### Run using Colab

Image capture

```python
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename
```

```python
from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))

  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
```

```python
Image('my_face.PNG')
```

```python
import cv2, dlib
import numpy as np
from imutils import face_utils
from keras.models import load_model

# import winsound (It is not used in Colab)



IMG_SIZE = (34, 26)
threshold_value_eye=0.4
count_frame= 0
Alarm_frame= 50

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eye_blink_detector/shape_predictor_68_face_landmarks.dat')

#change your model 
model = load_model('models/2021_06_20_08_37_01.h5')
# model.summary()
```

```python
#######crop eye######
def crop_eye(img, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.int)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect
#######crop eye#######
```

```python
# main
from google.colab.patches import cv2_imshow
img_ori = cv2.imread("photo.jpg") #capture imaage load 



img_ori = cv2.resize(img_ori, dsize=(600, 600), fx=0.5, fy=0.5)
img = img_ori.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## face detector 

faces = detector(gray)

img=cv2.flip(img,1) # Flip the image.
img_h, img_w,_ =img.shape #image width, height

for face in faces: 

  shapes = predictor(gray, face) # 68-point landmark detectors


  shapes = face_utils.shape_to_np(shapes) # shape to numpy

  sh=np.array(shapes) # numpy to array 

  #36-48 out of 68 points (eyes point)
  eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shapes[36:42]) 
  eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shapes[42:48])

  #resize 
  eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
  eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)

  #normalization 
  eye_input_l = eye_img_l.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.
  eye_input_r = eye_img_r.copy().reshape((1, IMG_SIZE[1], IMG_SIZE[0], 1)).astype(np.float32) / 255.

  #predict eye 
  pred_l = model.predict(eye_input_l)
  pred_r = model.predict(eye_input_r)

  # visualize (threshold_value_eye=0.4)
  state_l = 'O' if pred_l > threshold_value_eye else '-'
  state_r = 'O' if pred_r > threshold_value_eye else '-'




  l1=(int(img_w)-eye_rect_l[0],eye_rect_l[1]) # rectangle x1 , y1 (left) 
  l2=(int(img_w)-eye_rect_l[2],eye_rect_l[3]) # rectangle x2 , y2 (left)

  r1=(int(img_w)-eye_rect_r[0],eye_rect_r[1]) # rectangle x1 , y1 (right)
  r2=(int(img_w)-eye_rect_r[2],eye_rect_r[3]) # rectangle x2 , y2 (right)

  cv2.rectangle(img, pt1=l1, pt2=l2, color=(255,255,255), thickness=2)
  cv2.rectangle(img, pt1=r1, pt2=r2, color=(255,255,255), thickness=2)

  # Alarm (Alarm_frame=50)  

  if (pred_l<threshold_value_eye and pred_r<threshold_value_eye): #  if eyes close
        count_frame=count_frame+1      
  else: # eyes open (= count_frame initialization)
        count_frame=0

  if count_frame>Alarm_frame: # Alarm run
        string_sign="warning"
        cv2.putText(img, string_sign, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        # winsound.Beep(2400,1000) #sound  (It is not used in Colab)
        # winsound.SND_PURGE


  else: 
        string_sign="safe" 

        cv2.putText(img, string_sign, (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

  # eye state 
  cv2.putText(img,state_l, (int(img_w)-eye_rect_l[0]-20,eye_rect_l[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
  cv2.putText(img,state_r, (int(img_w)-eye_rect_r[0]-20,eye_rect_r[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

  # Facial Outlines
  sh=sh[0:27] 
  face_n=len(sh)

  for i in range(face_n):
      cv2.circle(img,(int(img_w)-sh[i][0],sh[i][1]),1,(0,0,255),1)

#show image 
cv2_imshow(img)
```

## VSCODE version

### 1. Setting up virtual Environment for Eye Blink Detector

#### anaconda environment create

```
conda create -n eye_blink python=3.6
```

#### github clone

```
git clone https://github.com/Jo951128/eye_blink_detector.git
```

#### library install

```
(Option 1)
pip install opencv-python
pip install cmake
pip install dlib
pip install imutils
conda install keras
pip install matplotlib
pip install sklearn

(Option 2)
pip install -r requirements.txt 
```

### 2. train

```
(TERMINAL command)
python .\train.py
```

#### CNN for regression prediction with keras

eye blink detector model

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FkHJlrrhXAysDF9GzL3ES%2Ffile.png?alt=media)

eye blink detector model result

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FwHbx3g8Qsw7iaNK8bF65%2Ffile.png?alt=media)

### 3. Run Webcam

```
(TERMINAL command)
python .\eye_blink.py
```

![image.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FgvYy7mwRc6H93S8ivkjw%2Ffile.png?alt=media)

## Reference

[https://github.com/kairess/eye\_blink\_detector](https://github.com/kairess/eye\_blink\_detector)

