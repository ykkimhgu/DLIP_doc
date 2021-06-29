# DLIP\_LAB4\_Report\_21500461\_이건호\_21600372\_송윤경

```text
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

