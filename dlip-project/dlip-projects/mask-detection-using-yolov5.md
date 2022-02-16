# Mask Detection using YOLOv5

**Date:** 2021-6-21

**Author**: 박성령, 김희락&#x20;

**Github:**&#x20;

## Introduction

Due to Covid-19, the social distancing level is elevated and gathering more than 5 people is prohibited in Korea. We designed this program in order to detect face mask and the gathering of people using deep learning and computer vision. We use YOLOv5 which is a pre-trained object detection model in Google Colab and Visual Studio Code through Anaconda virtual environment.

## 1. Download Image and Annotation files

### 1.1. Download Zip file

(Annotation file's format : XML)

![1.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FiWUfH4EEdeM6rlsNy4Kz%2Ffile.png?alt=media)

### 1.2. XML to Text(Yolov5) Conversion

Object detection by using YOLOv5 requires a compatible label format. Since the annotation downloaded from kaggle has a .xml format, we convert it to a .txt file according to the method presented.

This website provides data transformation method: [https://app.roboflow.com](https://app.roboflow.com)

Sing up with E-mail or Github for free.

![tuto\_1.PNG](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F5Pj45VV488yDIhiFdcGs%2Ffile.png?alt=media)

Skip tutorial and create new object detection project. Roboflow provides object detection type basically and you do not need to change.

![tuto\_3x.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FPgC4Ph2aGphSsiEpoeX0%2Ffile.png?alt=media)

Now, you can upload data to roboflow. Select the path to your images and annotations.

Then, the labels are indicated on the images in position.

![tuto\_6x.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FfqqHYG1USRnkvTlBzZrR%2Ffile.png?alt=media)

You can assign the proportion of training, testing and validation set. For this step, we continue as default set since we do not need to separate data set. (You can set all data to single set such as training, testing, or validation)

![tuto\_7x.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FO1eurUBZMZMiAsvWXeOR%2Ffile.png?alt=media)

You can preprocess the data such as 'Auto-Orient' and 'Resize' before converting. These process are recommanded but not essential and you can skip.

Click 'Generate' button on the bottom of the page and finish conversion.

![tuto\_9x.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FjIrOnEr2IVEgLeS9BY9A%2Ffile.png?alt=media)

Select YOLOv5 Pytorch data format and download.

![tuto\_10xx.PNG](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FBHfLXxHNAs2DKBKi64GN%2Ffile.png?alt=media)

## 2. Training

### 2.1. Recall converted image and annotation files

If you have run the code, you can see the images and label folders for train, test, and valid created in the file window on the left.

![KakaoTalk\_20210611\_141933313.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FARBO1425AhuHL1HbH0q7%2Ffile.png?alt=media)

```python
!curl -L "https://app.roboflow.com/ds/0nOwfb8NkY?key=rzt0i3M7lf" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

### 2.2. Recall YOLOv5 at github clone

![1231.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FdkmY2u3lXfaZSK1SF0lI%2Ffile.png?alt=media)

```python
%cd /content
!git clone https://github.com/ultralytics/yolov5.git
```

```
/content
Cloning into 'yolov5'...
remote: Enumerating objects: 6990, done.[K
remote: Counting objects: 100% (96/96), done.[K
remote: Compressing objects: 100% (93/93), done.[K
remote: Total 6990 (delta 48), reused 16 (delta 3), pack-reused 6894[K
Receiving objects: 100% (6990/6990), 9.10 MiB | 24.46 MiB/s, done.
Resolving deltas: 100% (4787/4787), done.
```

### 2.3. Install Packages for YOLOv5

```python
%cd /content/yolov5/
!pip install -r requirements.txt
```

### 2.4. Check that the yaml file

Check that the data set information is written correctly in the yaml file.

```python
 %cat /content/data.yaml
```

```
train: ../train/images
val: ../valid/images

nc: 3
names: ['mask_weared_incorrect', 'with_mask', 'without_mask']
```

### 2.5. Divide Dataset

Currently, the train and validation datasets are randomly divided. But we want to shuffle the data every time we train. So put all the data together and shuffle them randomly.

Create a list for all image files

```python
%cd /
from glob import glob
img_list = glob('/content/train/images/*.jpg') + glob('/content/test/images/*.jpg') + glob('/content/valid/images/*.jpg')
len(img_list)
```

```
/





1441
```

Divide the train set and validation set by a ratio of 0.25.

```python
from sklearn.model_selection import train_test_split
train_img_list, valid_img_list = train_test_split(img_list, test_size = 0.25, random_state = 2000)

print(len(train_img_list), len(valid_img_list))
```

```
1080 361
```

Save the training image file and the image path of the validation image file as .txt file

![asdasds.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FsdiXDX3zzyWqHBtWNJNz%2Ffile.png?alt=media)

```python
with open('/content/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('/content/valid.txt', 'w') as f:
  f.write('\n'.join(valid_img_list) + '\n')
```

```python
import yaml

with open('/content/data.yaml', 'r') as f:
  data = yaml.load(f)

print(data)

data['train'] = '/content/train.txt'
data['val'] = '/content/valid.txt'

with open('/content/data.yaml', 'w') as f:
  yaml.dump(data, f)

print(data)
```

```
{'train': '../train/images', 'val': '../valid/images', 'nc': 3, 'names': ['mask_weared_incorrect', 'with_mask', 'without_mask']}
{'train': '/content/train.txt', 'val': '/content/valid.txt', 'nc': 3, 'names': ['mask_weared_incorrect', 'with_mask', 'without_mask']}
```

### 2.6 Imageset Training

You can set the various parameters needed for training.

```python
%cd /content/yolov5/

!python train.py --img 416 --batch 16 --epochs 60 --data /content/data.yaml --cfg ./models/yolov5x.yaml --weights yolov5x.pt --name mask_yolov5_results
```

### 2.7 Training result

You can see the result of training at the tensorboard

### and You MUST save the weights(best.pt) at the left file-folder at your own computer

(yolov5 -> runs -> weights -> best.pt)

```python
%load_ext tensorboard
%tensorboard --logdir /content/yolov5/runs/
```

## 3. Run Demo Video by using VSCODE

### 3.1. Setting the YOLOv5 Training Environment

Create Anaconda Environment

![ana.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F6AX3OJDSnDtxdhQj3vM3%2Ffile.png?alt=media)

Install Required Packages

![package.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2F9p9dOPBAtGc39xXubNLZ%2Ffile.png?alt=media)

Download Yolov5 Zip\_file

[https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

![dlz.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FAKYmx6cclnfWmVxhQoes%2Ffile.png?alt=media)

run VScode

![run.png](https://firebasestorage.googleapis.com/v0/b/gitbook-x-prod.appspot.com/o/spaces%2F-MR8tEAjhiC8uN1kHR2J%2Fuploads%2FuSkJhc4FO32If6PYXFop%2Ffile.png?alt=media)

### 3.2. Customizing

Before modifying and executing the code, the previously downloaded weights file(best.pt) and test images must be placed in the yolov5 folder.

#### modifying detect.py

> Detecting how many people per class by frame

```python
                num_of_ic_mask = 0
                num_of_with_mask = 0
                num_of_wo_mask = 0

                # Print results
                for c in det[:, -1].unique():

                    n = (det[:, -1] == c).sum()  # detections per class             
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    classnum = c.item()

                    if classnum == 0:
                        num_of_ic_mask = n.item()

                    elif classnum == 1:
                        num_of_with_mask = n.item()

                    elif classnum == 2:
                        num_of_wo_mask = n.item()
```

#### Plot How many people per class by frame

```python
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format

                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integ er class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')                       

                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness)

                        total_person = (num_of_with_mask + num_of_wo_mask + num_of_ic_mask)
                        cv2.putText(im0, '# of detected person :' + str(total_person), (360, 50), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)
                        cv2.putText(im0, 'With Mask :' + str(num_of_with_mask), (487, 80), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)
                        cv2.putText(im0, 'Incorrect Mask :' + str(num_of_ic_mask), (435, 110), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)
                        cv2.putText(im0, 'Without Mask :' + str(num_of_wo_mask), (452, 140), cv2.FONT_ITALIC, 0.7, (0, 255, 0), 2)

                        # warning sign
                        if total_person > 4:
                            cv2.putText(im0, 'BAN gatherings of 5 or more people!!', (15, 450), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)

                        if opt.save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
```

#### modifying plots.py

**Color classification by class**

> incorrect\_wear\_mask : class\[0] - 'FFB330' (Orange)
>
> with\_mask : class\[1] - '0B6CE3' (Blue)
>
> without\_mask : class\[2] - 'FF0000' (Red)

```python
class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FFB330', '0B6CE3', 'FF0000', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))
```

RUN!

```python
python detect.py --source .\mask_video.mp4 --weights .\best_x.pt --view --conf-thres 0.2 --iou-thres 0.6 --agnostic-nms
```
