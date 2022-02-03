# Digital Door Lock Control with Face Recognition

**Date:** 2021-6-21

**Author**: Keunho Lee, Yoonkyoung Song

**Github:** [https://github.com/DLIP-Digital-Doorlock-Control-Deep/Digital\_Doorlock\_Control\_Face\_Recognition](https://github.com/DLIP-Digital-Doorlock-Control-Deep/Digital\_Doorlock\_Control\_Face\_Recognition)

## **Introduction**

In this LAB, we started a project to learn human faces and control only authorized users to open digital door locks. After creating an authorized user's dataset to learn about the face, the digital door lock is opened for registered users. To achieve this in an embedded environment, a real-time door lock can be controlled using a Web Cam, Jetson Nano, Arduino UNO, and digital door lock. In addition, the OpenCV library was used to find and learn faces processed by software.

The full version of this project Code is in the next following github URL.

[https://github.com/DLIP-Digital-Doorlock-Control-Deep/Digital\_Doorlock\_Control\_Face\_Recognition](https://github.com/DLIP-Digital-Doorlock-Control-Deep/Digital\_Doorlock\_Control\_Face\_Recognition)

### **Hardware Setting**

![3-1.JPG](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FnhJlnXdCnHDZwFu1BaGj%2Ffile.jpeg?alt=media)

To build hardware, video cam, Jetson Nano, Arduino UNO, and digital door lock are used. The hardware connection is shown in the figure above. First, we receive frames from where the camera is located through a video cam and search for faces in the image on Jetson Nano. When an authorized user's face is recognized, it sends a command to Arduino via serial communication. Arduino is connected to the digital door lock. When Arduino is ordered to receive signals from Jetson Nano, Arduino command digital door locks to open.

### **Software Processes**

![4.jpg](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2Fkl4jm1oZO1pDcfbx5A1Q%2Ffile.jpeg?alt=media)

### System summary

**In this project, it is a system that trains the faces of people who have registered in advance (in this project, two pre-registered authorized users), and sends a command to open the door if the similarity with the faces of the people who have learned in advance is higher than the threshold.**

## **Software Explanation**

### + _Face Detection Using OpenCV Harr Cascade Detection Algorithm_

Haar Cascade Detection Algorithm is an algorithm that detects a specific object in an image. In this project, it has used to detect human faces in an image.

This algorithm is an algorithm that finds a specific object in an image by learning the pattern for the characteristics of the object. This algorithm is also frequently used for face detection. The algorithm of Haar Cascade is simply described as follows.

* First, there are similar features exist in human faces. When we call this a Haar like feature for human face, the key is to recognize it as a face when the threshold is exceeded for each feature.
* Adaboost learning algorithm is a simple and efficient classifier to select a small number of critical visual features from a very large set of potential features.
* Cascade classifier algorithm configures all cascades for the Haar cascade and detects the face object to be found only when the entire cascade has agreed about the feature.

In OpenCV, it offers "haarcascade\_frontalface\_default.xml" file, which is pretrained xml file for frontal face.

The face found by Harr Cascade Detection Algorithm, now goes next stage to extract the characteristics of each person by LPBH algorithm

### + _Local Binary Patterns Histogram(LBPH) Algorithm_

The LBPH algorithm stands for Local Binary Patterns Histogram, which literally means a filter with a specific size is applied to the image. After applying a threshold to the brightness of each pixel, it is classified into binary, and then a histogram is drawn by calculating the pattern between the binaries in the filter. In this way, the human face can be divided into several sections and the histogram value can be calculated in each section. If these histogram values ​​are similar, it is judged as the same person.

**Part 1. Creating Authorized Users Face Dataset**

#### Detect a face and crop a face from the users face dataset images. Save the cropped image as a 200x200 size gray image.

```python
import cv2
import numpy as np
import os

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face


Face_ID = -1 
pev_person_name = ""

Face_Images = os.path.join(os.getcwd(), "Face_Images") #이미지 폴더 지정
print (Face_Images)

for root, dirs, files in os.walk(Face_Images) : #파일 목록 가져오기
    for file in files :
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png") : #이미지 파일 필터링
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, person_name)

            if pev_person_name != person_name : #이름이 바뀌었는지 확인
                Face_ID=Face_ID+1
                pev_person_name = person_name
                count = 0

            img = cv2.imread(path) #이미지 파일 가져오기

            if face_extractor(img) is not None:
                count+=1
                face = cv2.resize(face_extractor(img),(200,200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = 'faces/'+person_name+'/'+person_name+str(count)+'.jpg'
                cv2.imwrite(file_name_path,face)

            else:
                print("Face not Found")
                pass

print('Colleting Samples Complete!!!')
```

```
/content/Face_Images
Colleting Samples Complete!!!
```

#### Part 2. Training Authorized Users Faces

```python
import cv2
import numpy as np
import os
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create() #LBPH를 사용할 새 변수 생성

Face_ID = -1 
pev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "faces") #이미지 폴더 지정
print (Face_Images)

for root, dirs, files in os.walk(Face_Images) : #파일 목록 가져오기
    for file in files :
        if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png") : #이미지 파일 필터링
            path = os.path.join(root, file)
            person_name = os.path.basename(root)
            print(path, person_name)

            if pev_person_name != person_name : #이름이 바뀌었는지 확인
                Face_ID=Face_ID+1
                pev_person_name = person_name

            img = cv2.imread(path) #이미지 파일 가져오기
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5) #얼굴 찾기

            print (Face_ID, faces)

            for (x,y,w,h) in faces:
                roi = gray_image[y:y+h, x:x+w] #얼굴부분만 가져오기
                x_train.append(roi)
                y_ID.append(Face_ID)

                recognizer.train(x_train, np.array(y_ID)) #matrix 만들기
                recognizer.save("face-trainner.yml") #저장하기
```

```
/content/faces
```

#### Part 3. Real-Time Digital Door Lock Control

**Real-time user recognizing code using web cam and save the output video. If you don't have arduino and just want to run this code, you just comment out of the 8 to 11 lines and 77 line, which is related  with running the Arduino.**

**When you want to run in window with arduino, just change the line number 8 to like 'COM3', which corresponds to the arduino port number.**

```python
import cv2
import numpy as np
import os
from PIL import Image
import serial
import time

PORT = '/dev/ttyACM0'             #When you want to run in window with arduino, just change this to like 'COM3', which corresponds to the arduino port number
BaudRate = 115200
detectTime =0

arduino = serial.Serial(PORT,BaudRate)
serialTime = time.time()

labels = ["Keunho", "Yoonkyoung"] #라벨 지정

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml") #저장된 값 가져오기

cap = cv2.VideoCapture(0) #카메라 실행

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 cap.get(3)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 cap.get(4)
fps = cap.get(cv2.CAP_PROP_FPS) # 또는 cap.get(5)
print('프레임 너비: %d, 프레임 높이: %d, 초당 프레임 수: %d' %(width, height, fps))

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('face_recog.avi', fourcc, fps, (int(width), int(height)))


if cap.isOpened() == False : #카메라 생성 확인
    exit()

while True :
    ret, img = cap.read() #현재 이미지 가져오기
    img = cv2.flip(img, 1)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #흑백으로 변환
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) #얼굴 인식
    result_conf = 0
    result_id_=""
    result_x=0
    result_y=0
    result_w=0
    result_h=0

    try:

        for (x, y, w, h) in faces :
            roi_gray = gray[y:y+h, x:x+w] #얼굴 부분만 가져오기
            #roi_gray = cv2.resize(roi_gray, (200,200))

            id_, conf = recognizer.predict(roi_gray) #얼마나 유사한지 확인
            print(id_, conf)

            if result_conf < conf:
                result_conf = conf
                result_id_=id_
                result_x=x
                result_y=y
                result_w=w
                result_h=h

        result_conf = int(100*(1-(result_conf)/300))

        if result_conf>=75:
            font = cv2.FONT_HERSHEY_SIMPLEX #폰트 지정
            name = labels[result_id_] + f"{result_conf}" #ID를 이용하여 이름 가져오기
            cv2.putText(img, name, (result_x,result_y), font, 1, (250,120,255), 2)
            cv2.rectangle(img,(result_x,result_y),(result_x+result_w,result_y+result_h),(0,255,255),2)
            cv2.putText(img, "Unlocked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            detectTime = time.time()-serialTime

            if (detectTime > 15):
                cmd = "1"
                cmd = cmd.encode('utf-8')
                arduino.write(cmd)
                serialTime= time.time()
        else:
            cv2.putText(img, "Locked", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Preview',img) #이미지 보여주기

    except:
        cv2.putText(img, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Preview', img)
        pass  

    out.write(img)
    if cv2.waitKey(10) >= 0: #키 입력 대기, 10ms
        break

#전체 종료
cap.release()
out.release()
cv2.destroyAllWindows()
```

### Result

* Real-time Face Recognition with embeded digital door lock system

### Evaluation

Although the purpose of the digital door lock system is high accuracy recognition of authorized users, however, the most important point in terms of security is to prevent unauthorized persons from being recognized as authorized persons. We raise the confidence threshold in the program so that even if the digital door lock will not be opened every time when authorized person try to open the door lock system,but the unauthorized person is not falsely detected as the authorized user.

[![confus.png](https://i.postimg.cc/6QT2bSCQ/confus.png)](https://postimg.cc/7C8ZfKsv)

The precision, recall, and accuracy from the confusion matrix of this system are as follows.

(Result from the demo videos, checking every single frame by manual. Two demo videos are used.)

* Accuracy = 60%
* Precision = 100%
* Recall = 37.2%

In the case of Accuracy, among all cases of judgment, it is the percentage of all correct answers, which means judging the unauthorized person as an unauthorized person and the authorized person as an authorized person.

In the case of Precision, it is the percentage of actually judged an authorized person out of all cases determined to be an authorized person. If the precision is not high, an unauthorized person may also be mistakenly recognized as an authorized person..

Recall is an indicator of how much a person is recognized to be an authorized person in a situation where they are actually authorized persons. If the recall is not high, the recognition speed will not be fast in the Real-Time system.

The most critical metric from our point of view was Precision. Since even a single false detection can be quite vulnerable to security, we tried to make the precision as close to 100% as possible.

### Reference

**Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Proceedings of the 2001 IEEE computer society conference on computer vision and pattern recognition. CVPR 2001. Vol. 1. IEEE, 2001.**

**Viola, Paul, and Michael J. Jones. "Robust real-time face detection." International journal of computer vision 57.2 (2004): 137-154**

**Haar Cascade Detection Algoritm**

[https://docs.opencv.org/4.1.0/dc/d88/tutorial\_traincascade.html](https://docs.opencv.org/4.1.0/dc/d88/tutorial\_traincascade.html)

#### Code Reference

* [http://blog.naver.com/PostView.nhn?blogId=chandong83\&logNo=221436424539\&parentCategoryNo=\&categoryNo=44\&viewDate=\&isShowPopularPosts=true\&from=search](http://blog.naver.com/PostView.nhn?blogId=chandong83\&logNo=221436424539\&parentCategoryNo=\&categoryNo=44\&viewDate=\&isShowPopularPosts=true\&from=search)
* [https://blog.naver.com/roboholic84/221533459586](https://blog.naver.com/roboholic84/221533459586)
* [https://github.com/AzureWoods/faceRecognition-yolo-facenet](https://github.com/AzureWoods/faceRecognition-yolo-facenet)

```python
```
