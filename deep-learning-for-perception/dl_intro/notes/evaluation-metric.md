# Evaluation Metric

## **ROC Curve**

Understanding AUC, ROC curve: [click here](https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5)

AUC - ROC curve is a performance measurement for classification problem at various thresholds settings. ROC is a probability curve and AUC represents degree or measure of separability.

ROC AUC It tells how much model is capable of distinguishing between binary classes

Receiver Operating Characteristic\(ROC\) Curve

* true positive rate \(recall\) vs false positive rate \(FPR\)
* FPR is the ratio of negative instances that are incorrectly classified as positive

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28261%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28255%29.png)

Area under the curve\(AUC\): a perfect classifier ROC AUC= 1 a purely random classifier ROC AUC= 0.5.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28259%29.png)

* E.g. Find a Person. Red: Person, Green: non-person

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28258%29.png)

## Top-1, TOP-5 ImageNet,  ILSVRC

The Top-5 error rate is the percentage of test examples for which the correct class was not in the top 5 predicted classes.

If a test image is a picture of a `Persian cat`, and the top 5 predicted classes in order are `[Pomeranian (0.4), mongoose (0.25), dingo (0.15), Persian cat (0.1), tabby cat (0.02)]`, then it is still treated as being 'correct' because the actual class is in the top 5 predicted classes for this test image.

## For Object Detection

### IOU

We need to evaluate the performance of both \(1\) classification and \(2\) localization of using bounding boxes in the image.

Object Detection uses the concept of **Intersection over Union \(IoU\)**. IoU computes intersection over the union of the two bounding boxes; the bounding box for the ground truth and the predicted bounding box. An IoU of 1 implies that predicted and the ground-truth bounding boxes perfectly overlap.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28270%29.png)

Set a threshold value for the IoU to determine if the object detection is valid or not.

If threshold of IoU=0.5,

* **if IoU ≥0.5,** classify the object detection as **True Positive\(TP\)**
* **if Iou &lt;0.5**, then it is a wrong detection and classify it as **False Positive\(FP\)**
* **When a ground truth is present in the image and model failed to detect the object,** classify it as **False Negative\(FN\).**
* **True Negative \(TN**\): TN is every part of the image where we did not predict an object. This metrics is not useful for object detection, hence we ignore TN.

Also, need to consider the **confidence score \(classification\)** for each object detected. Bounding boxes above the threshold value are considered as _positive_ boxes and all predicted bounding boxes below the threshold value are considered as _negative_.

Use Precision and Recall as the metrics to evaluate the performance. Precision and Recall are calculated using true positives\(TP\), false positives\(FP\) and false negatives\(FN\).

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28260%29.png)

### mAP

> What is mAP: [click here](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)

It use 11-point interpolated average precision to calculate mean Average Precision\(mAP\).

**Step 1: Plot Precision and Recall from IoU**

> Precision in PR graph is not always monotonically decreasing due to certain exceptions and/or lack of data.

Example: In this example, the whole dataset contains 5 apples only. We collect all the predictions made for apples in all the images and rank it in descending order according to the predicted confidence level. \(IoU&gt;0.5\)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28256%29.png)

> For example, for rank\#3, assume only 3 apples are predicted\(2 are correct\)
>
> **Precision** is the proportion of TP = 2/3 = 0.67
>
> **Recall** is the proportion of TP out of the possible positives = 2/5 = 0.4

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28265%29.png)

**Step 2: use 11 point interpolation technique.**

11 equally spaced recall levels of 0.0, 0.1, 0.2, 0.3 ….0.9, 1.0.

Point interpolation: take the maximum Precision value of all future points of Recall.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28257%29.png)

**Step 3: Calculate the mean Average Precision\(mAP\)**

Average Precision is the area under the curve of Precision-Recall

mAP is calculated as

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28266%29.png)

In our example, AP = \(5 × 1.0 + 4 × 0.57 + 2 × 0.5\)/11.

For 20 different classes in PASCAL VOC, we compute an AP for every class and also provide an average for those 20 AP results.

It is less precise. Second, it lost the capability in measuring the difference for methods with low AP. Therefore, a different AP calculation is adopted after 2008 for PASCAL VOC.

### AP \(Area under curve AUC\)

For later Pascal VOC competitions, VOC2010–2012 samples.

No approximation or interpolation is needed. Instead of sampling 11 points, we sample _p_\(_rᵢ_\) whenever it drops and computes AP as the sum of the rectangular blocks.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28264%29.png)

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28267%29.png)

### **COCO mAP**

Latest research papers tend to give results for the COCO dataset only. In COCO mAP, a 101-point interpolated AP definition is used in the calculation. For COCO, AP is the average over multiple IoU \(the minimum IoU to consider a positive match\). **AP@\[.5:.95\]** corresponds to the average AP for IoU from 0.5 to 0.95 with a step size of 0.05. For the COCO competition, AP is the average over 10 IoU levels on 80 categories \(AP@\[.50:.05:.95\]: start from 0.5 to 0.95 with a step size of 0.05\). The following are some other metrics collected for the COCO dataset.

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28271%29.png)

