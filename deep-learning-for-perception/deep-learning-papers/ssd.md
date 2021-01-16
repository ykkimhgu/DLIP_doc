# SSD

## Introduction

[SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) \(by W.Liu, C. Szegedy et al.,2016\), object detector scoring over 74% mAP at 59 FPS on [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/) and [COCO](http://cocodataset.org/#home).

* **Single Shot:** this means that the tasks of object localization and classification _\_are done in a \_single_ _forward pass_ of the network
* **MultiBox:** this is the name of a technique for bounding box regression developed by Szegedy et al. \(we will briefly cover it shortly\)
* **Detector:** The network is an object detector that also classifies those detected objects

![](../../.gitbook/assets/image%20%28204%29.png)

## Single Shot MultiBox Detector <a id="7a47"></a>

### Architecture

Based on VGG-16. No FC layers but added a set of _auxiliary_ convolutional layers \(from _conv6_ onwards\). This can extract features at multiple scales and progressively decrease the size of the input to each subsequent layer.

![A comparison between two single shot detection models: SSD and YOLO \[5\]. Our SSD model adds several feature layers to the end of a base network, which predict the offsets to default boxes of different scales and aspect ratios and their associated confidences. SSD with a 300 &#xD7; 300 input size significantly outperforms its 448 &#xD7; 448 YOLO counterpart in accuracy on VOC2007 test while also improving the speed.](../../.gitbook/assets/image%20%28203%29.png)

#### Baseline: VGG architecture

![VGG architecture \(input is 224x224x3\)](../../.gitbook/assets/image%20%28205%29.png)

#### MultiBox

The bounding box regression technique of SSD is inspired by Szegedy’s work on [MultiBox](https://arxiv.org/abs/1412.1441). MultiBox starts with the **priors as predictions** and attempt to regress closer to the ground truth bounding boxes. At the end, MultiBox only retains the top K predictions that have minimised both location \(_LOC_\) and confidence \(_CONF_\) losses.

Multibox: It is for Bounding Box Proposal of fast _class-agnostic_ bounding box coordinate proposals.

> Class-agnostic means bounding box of object without classfication

Multibox contains 11 priors per feature map cell \(8x8, 6x6, 4x4, 3x3, 2x2\) and only one on the 1x1 feature map, resulting in a total of 1420 priors per image

> 11\*\(8\*8\)+11\*\(6\*6\)+...+11\*\(2\*2\)=1419 + 1\(1\*1\)=1420

![Architecture of multi-scale convolutional prediction of the location and confidences of multibox](../../.gitbook/assets/image%20%28210%29.png)

MultiBox’s loss function also combined two critical components:

_**multibox\_loss = confidence\_loss + alpha \* location\_loss**_

![](../../.gitbook/assets/image%20%28208%29.png)

* **Confidence Loss**: this measures how confident the network is of the _objectness_ of the computed bounding box. Categorical [cross-entropy](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/#cross-entropy) is used to compute this loss.
* **Location Loss:** this measures how _far away_ the network’s predicted bounding boxes are from the ground truth ones from the training set. [L2-Norm](https://rorasa.wordpress.com/2012/05/13/l0-norm-l1-norm-l2-norm-l-infinity-norm/) is used here.

![](../../.gitbook/assets/image%20%28209%29.png)

![](../../.gitbook/assets/image%20%28207%29.png)

### **Results** <a id="ce75"></a>

#### **Default Bounding Boxes**

The SSD paper has around 6 bounding boxes per feature map cell.

#### **Pascal VOC 2007**

![](../../.gitbook/assets/image%20%28211%29.png)

## Additional Notes On SSD <a id="52ce"></a>

The SSD paper makes the following additional observations:

* more default boxes results in more accurate detection, although there is an impact on speed
* having MultiBox on multiple layers results in better detection as well, due to the detector running on features at multiple resolutions
* 80% of the time is spent on the base VGG-16 network: this means that with a faster and equally accurate network SSD’s performance could be even better
* SSD confuses objects with similar categories \(e.g. animals\). This is probably because locations are shared for multiple classes
* SSD-500 \(the highest resolution variant using 512x512 input images\) achieves best mAP on Pascal VOC2007 at 76.8%, but at the expense of speed, where its frame rate drops to 22 fps. SSD-300 is thus a much better trade-off with 74.3 mAP at 59 fps.
* SSD produces worse performance on smaller objects, as they may not appear across all feature maps. Increasing the input image resolution alleviates this problem but does not completely address it

