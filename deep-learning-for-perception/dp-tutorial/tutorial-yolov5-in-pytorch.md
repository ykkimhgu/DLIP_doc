# Tutorial: Yolov5 in PyTorch

**A Simple Run YOLOv5 example in Local PC **

- 본 튜토리얼에서는 local PC에 설치한 `py39`환경으로 YOLO v5를 실행하는 두가지 방법의 간단한 예제를 제공합니다.

  1) 명령창을 활용해 구동하는 방법

  2) VS code와 같은 IDE로 torch hub를 활용하여 구동하는 방법

- YOLO v5에 대한 구조적 설명 및 이론적인 배경은 제공하지 않습니다.

- 반드시 [Installation Guide for DLIP](https://ykkim.gitbook.io/dlip/installation-guide/installation-guide-for-deep-learning) 을 먼저 완료한 후 실행하십시오



# Preparation (필요한 Libs 설치)

YOLOv5 github(https://github.com/ultralytics/yolov5)에 접속하여 아래와 같이 다운로드합니다. 

![image](https://user-images.githubusercontent.com/23421059/169227977-bf94857e-3e87-4cc5-9d1d-daf73836a3dd.png)



압축해제 후 파일명 yolov5-master → yolov5로 변경 및 원하는 곳에 폴더 붙여넣습니다.

yolov5 폴더에 진입 후 아래 그림과 같이 경로 주소를 복사(ctrl+C)합니다.

![image](https://user-images.githubusercontent.com/23421059/169229474-723ba3ae-2c70-4bcf-8d4d-760543c79fb1.png)



Anaconda prompt를 관리자모드로 실행 후 아래 코드 순차적으로 실행합니다 (아래 그림 참조)

```
conda activate py39
cd [ctrl+V] // 복사한 yolov5 경로 붙여넣기 (그림 참조)
pip install -r requirements.txt
```

![image](https://user-images.githubusercontent.com/23421059/169230206-55eacf01-0b72-42a2-b8c2-2b046572d5bb.png)



이후 설치가 완료됩니다. 아래와 같이 경고가 뜨지만 괜찮습니다.

![image](https://user-images.githubusercontent.com/23421059/169255844-7db4db53-9129-41be-a4f3-8f395b369c83.png)





# Run YOLOv5 in Local PC with CLI

- 명령창(command line, CLI)로 YOLO v5 실행시 `detect.py`, `val.py`, `train.py`와 같이 git에서 제공된 파일을 빌드합니다.
- 따라서 CLI로 빌드시, 반드시 git을 local PC에 저장하고 폴더 경로에 진입하는 과정이 선행되어야 합니다.



## 1. Inference

yolov5 폴더 경로에 진입한 상태에서 아래 코드 입력하여 `detect.py`를 실행합니다.

```
python detect.py --weights yolov5n.pt --img 640 --conf 0.25 --source data/images
```



결과창. 아래와 같이 객체검출 저장된 폴더가 상대경로로 표시됩니다. 

![image](https://user-images.githubusercontent.com/23421059/169257427-4450a074-18d0-48a7-aec7-ffda79cda7c2.png)



실제로 해당 상대경로로 진입하면 결과를 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/23421059/169253425-211189e7-c537-490c-8454-699bc5617ad5.png)



## 2. Train

yolov5 폴더 경로가 유지된 상태에서 아래 코드 입력하여 `train.py`를 실행합니다.

```
python train.py --img 640 --batch 1 --epochs 1 --data coco128.yaml --weights yolov5n.pt
```



결과창. 아래와 같이 학습결과가 저장된 폴더가 상대경로로 표시됩니다. 

![image](https://user-images.githubusercontent.com/23421059/169253960-30e810ec-7c90-4602-94fc-b4df96ae7c80.png)



실제로 해당 상대경로로 진입하면 결과를 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/23421059/169254257-3636431b-3cc1-4b39-bfcf-78d282256f4d.png)





# Run YOLOv5 in Local PC with PyTorch Hub

- VS code와 같은 IDE에서도 YOLO v5 결과에 접근하여 프로그래밍할 수 있습니다.
- 필요한 모듈(requirements)만 설치하면, git을 local PC에 저장하는 선행과정이 필요하지 않습니다.
- [######]을 참고하여 VS code 설치 및 사용법 숙지를 선행하시기 바랍니다.



## Inference

임의의 폴더를 생성 후 우클릭 → Code로 열기를 클릭합니다.

![](https://user-images.githubusercontent.com/23421059/169258661-a30f94a3-96b9-4890-9a9d-7c4eb5aea4f8.png)



새파일을 만들고 아래의 코드를 붙여 넣습니다.

```python
import torch
import cv2
import random
from PIL import Image

# Load the Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

# Image preparation from URL . images link
img_URL = [           
           "https://user-images.githubusercontent.com/23421059/168719874-be48ef28-954c-4a4c-a048-1e11699e0b56.png",
           ]

imgs = []
img_L = len(img_URL)

# Append these 3 images and save 
for i in range(img_L):
  imgName = f"{i}.jpg"
  torch.hub.download_url_to_file(img_URL[i],imgName)  # download 2 images
  # imgs.append(Image.open(fileName))  # PIL image
  imgs.append(cv2.imread(imgName)[:,:,::-1]) # OpenCV image (BGR to RGB)

# Run Inference
results = model(imgs)

# Print Results
results.print()

# Save Result images with bounding box drawn
results.save()  # or .show()

# Select a random test image
randNo = random.choice(range(img_L))
print(f"Selected Image No = {randNo}\n\n")

# Print the Bounding Box result:  6 columns
# Column (1~4) Coordinates of TL, BR corners (5) Confidence (6) Class ID
print(results.xyxy[randNo],'\n')  # imgs predictions (tensor)

# Print the Bounding Box result using Pandas
print(results.pandas().xyxy[randNo],'\n')  # imgs predictions (pandas)

# Show result image
cv2.imshow("result", (results.imgs[randNo])[:,:,::-1])
cv2.waitKey(0)
```



아래 그림과 같이 위 코드가 새 python 파일에 작성되었습니다.

![image](https://user-images.githubusercontent.com/23421059/169262592-30479afb-298c-472a-ade6-8483939c3bbb.png)



`F1`키를 눌러 `select interpreter`를 검색 후 클릭 → `py39`를 선택합니다.

![image](https://user-images.githubusercontent.com/23421059/169260982-d5dc20a9-9cc4-4b63-8fd1-db3841323358.png)



`F1`키를 눌러 `select default profile`을 검색 후 클릭 → `command prompt`를 선택합니다.

![image](https://user-images.githubusercontent.com/23421059/169261544-f5b2d98a-5e0f-49f0-9e19-2e5a75c705ba.png)



`F5` 또는 `ctrl+F5`를 눌러 빌드하여 아래와 같이 결과창이 뜨는 것을 확인합니다

![image](https://user-images.githubusercontent.com/23421059/169261821-7f6dc614-1dd3-44ff-946a-55729107e348.png)



아래와 같이 학습결과가 저장된 폴더가 상대경로로 표시됩니다. 

![image](https://user-images.githubusercontent.com/23421059/169266835-1b4d1ce7-f70a-4d87-b968-215b51cc21ee.png)



실제로 해당 상대경로로 진입하면 결과를 확인할 수 있습니다.

![image](https://user-images.githubusercontent.com/23421059/169267297-9ea714d3-a6eb-4304-a634-103df5d76fde.png)