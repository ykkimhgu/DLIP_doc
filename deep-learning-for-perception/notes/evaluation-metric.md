# Evaluation Metric

## **ROC Curve**

Understanding AUC, ROC curve: [click here](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability.

ROC AUC It tells how much model is capable of distinguishing between binary classes

Receiver Operating Characteristic(ROC) Curve

* true positive rate (recall) vs false positive rate (FPR)
* FPR is the ratio of negative instances that are incorrectly classified as positive

![](<../../.gitbook/assets/image (261).png>)

![](<../../.gitbook/assets/image (255).png>)

Area under the curve(AUC): a perfect classifier ROC AUC= 1 a purely random classifier ROC AUC= 0.5.

![](<../../.gitbook/assets/image (259).png>)

* E.g. Find a Person. Red: Person, Green: non-person

![](<../../.gitbook/assets/image (258).png>)

### 민감도와 특이 (Covid-19 진단예시)

앞서 밝힌대로 민감도와 특이도 검사 모두 이미 음성·양성을 확인한 대상자를 놓고 새로운 진단법에 대한 정확도를 밝히는 과정이다.

민감도는 '양성 환자 중 검사법이 진단한 양성 정확도'라는 의미고, 특이도는 '정상인 중 검사법이 진단한 정상 정확도'라는 의미다.

실제 양성·음성군을 대상으로 진단 시행 시 얻을 수 있는 결과

각 표본에 대한 검사가 끝나면 대상군은 위의 네개로 분류되고 이때의 민감도·특이도를 구하는 공식은 다음과 같다.

![](http://cdn.hitnews.co.kr/news/photo/202010/30562\_30782\_1435.png)

1.  민감도 = 새로운 진단법이 판명한 환자 중 실제 환자

    ① / ① + ②
2.  특이도 = 새로운 진단법이 판명한 정상인 중 실제 정상

    ④ / ③ + ④

그렇다면 민감도와 특이도 중 진단기법 신뢰도에 더 큰 영향을 미치는 것은 무엇일까. 전문가에 따르면 두 기준은 양립해야하며 어느 한 쪽이 우월한 가치는 아니라는 설명이다.

정은경 청장은 "Sensitivity(민감도)와 Specificity(특이도)가 차이가 크다면 올바른 진단 방식이라고 볼 수는 없을 것"이라며 "질병에 따라 어느 한쪽에 무게가 실리기도 하지만 양쪽 모두를 충족해야 한다"고 설명했다.

출처 : 히트뉴스([http://www.hitnews.co.kr](http://www.hitnews.co.kr))

##

## Top-1, TOP-5 ImageNet, ILSVRC

The Top-5 error rate is the percentage of test examples for which the correct class was not in the top 5 predicted classes.

If a test image is a picture of a `Persian cat`, and the top 5 predicted classes in order are `[Pomeranian (0.4), mongoose (0.25), dingo (0.15), Persian cat (0.1), tabby cat (0.02)]`, then it is still treated as being 'correct' because the actual class is in the top 5 predicted classes for this test image.

## For Object Detection

### IOU

We need to evaluate the performance of both (1) classification and (2) localization of using bounding boxes in the image.

Object Detection uses the concept of **Intersection over Union (IoU)**. IoU computes intersection over the union of the two bounding boxes; the bounding box for the ground truth and the predicted bounding box. An IoU of 1 implies that predicted and the ground-truth bounding boxes perfectly overlap.

![](<../../.gitbook/assets/image (270).png>)

Set a threshold value for the IoU to determine if the object detection is valid or not.

If threshold of IoU=0.5,

* **if IoU ≥0.5,** classify the object detection as **True Positive(TP)**
* **if Iou <0.5**, then it is a wrong detection and classify it as **False Positive(FP)**
* **When a ground truth is present in the image and model failed to detect the object,** classify it as **False Negative(FN).**
* **True Negative (TN**): TN is every part of the image where we did not predict an object. This metrics is not useful for object detection, hence we ignore TN.

Also, need to consider the **confidence score (classification)** for each object detected. Bounding boxes above the threshold value are considered as _positive_ boxes and all predicted bounding boxes below the threshold value are considered as _negative_.

Use Precision and Recall as the metrics to evaluate the performance. Precision and Recall are calculated using true positives(TP), false positives(FP) and false negatives(FN).

![](<../../.gitbook/assets/image (260).png>)

### mAP

> What is mAP: [click here](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

It use 11-point interpolated average precision to calculate mean Average Precision(mAP).

**Step 1: Plot Precision and Recall from IoU**

> Precision in PR graph is not always monotonically decreasing due to certain exceptions and/or lack of data.

Example: In this example, the whole dataset contains 5 apples only. We collect all the predictions made for apples in all the images and rank it in descending order according to the predicted confidence level. (IoU>0.5)

![](<../../.gitbook/assets/image (256).png>)

> For example, for rank#3, assume only 3 apples are predicted(2 are correct)
>
> **Precision** is the proportion of TP = 2/3 = 0.67
>
> **Recall** is the proportion of TP out of the possible positives = 2/5 = 0.4

![](<../../.gitbook/assets/image (265).png>)

**Step 2: use 11 point interpolation technique.**

11 equally spaced recall levels of 0.0, 0.1, 0.2, 0.3 ….0.9, 1.0.

Point interpolation: take the maximum Precision value of all future points of Recall.

![](<../../.gitbook/assets/image (257).png>)

**Step 3: Calculate the mean Average Precision(mAP)**

Average Precision is the area under the curve of Precision-Recall

mAP is calculated as

![](<../../.gitbook/assets/image (266).png>)

In our example, AP = (5 × 1.0 + 4 × 0.57 + 2 × 0.5)/11.

For 20 different classes in PASCAL VOC, we compute an AP for every class and also provide an average for those 20 AP results.

It is less precise. Second, it lost the capability in measuring the difference for methods with low AP. Therefore, a different AP calculation is adopted after 2008 for PASCAL VOC.

### AP (Area under curve AUC)

For later Pascal VOC competitions, VOC2010–2012 samples.

No approximation or interpolation is needed. Instead of sampling 11 points, we sample _p_(_rᵢ_) whenever it drops and computes AP as the sum of the rectangular blocks.

![](<../../.gitbook/assets/image (264).png>)

![](<../../.gitbook/assets/image (267).png>)

### **COCO mAP**

Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU (the minimum IoU to consider a positive match). **AP@\[.5:.95]** corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories (AP@\[.50:.05:.95]: start from 0.5 to 0.95 with a step size of 0.05). The following are some other metrics collected for the COCO dataset.

![](<../../.gitbook/assets/image (271).png>)
