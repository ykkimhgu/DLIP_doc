---
description: Selected Papers and blogs for Perception Deep Learning
---

# Must Read Papers

### Must read papers on CNN

[A brief note on these selected papers: ](https://docs.google.com/document/d/1oTWU1kJXEOEvWUgKN878kvx3wSvjzk9aVXTUyABxDhM/edit?usp=sharing)

#### Popular Backbone for CNN

* [Alexnet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf):  [review on alexnet](https://medium.com/coinmonks/paper-review-of-alexnet-caffenet-winner-in-ilsvrc-2012-image-classification-b93598314160)
* [VGGNet](https://arxiv.org/pdf/1409.1556):  [review on VGGNet](https://medium.com/coinmonks/paper-review-of-vggnet-1st-runner-up-of-ilsvlc-2014-image-classification-d02355543a11)
* [ResNet](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) 2015
* Inception:  [ simple guide to Inception](https://towardsdatascience.com/a-simple-guide-to-the-versions-of-the-inception-network-7fc52b863202)
  * v1 2014: GoogLeNet 
  * [v3: 2015,](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf)   [review on v3](https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c)
  * [v4: paper,](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14806/14311)  review on v4
* Feature Pyramid Network:  [review on FPN ](https://towardsdatascience.com/review-fpn-feature-pyramid-network-object-detection-262fc7482610)

#### Object Detection

1. Two-Stage Detector

* R-CNN:  [reading on R-CNN and variants](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)
  * [R-CNN 2013](http://islab.ulsan.ac.kr/files/announcement/513/rcnn_pami.pdf) ,  [review on RCNN](https://medium.com/coinmonks/review-r-cnn-object-detection-b476aba290d1)
  * [Fast R-CNN: 2015](https://arxiv.org/abs/1504.08083)
  * [Faster R-CNN: 2015](https://arxiv.org/abs/1506.01497)

2.One-Stage Detector

* YOLO
  * [v1 2016:](https://arxiv.org/pdf/1506.02640v5.pdf)  [v2](https://arxiv.org/abs/1612.08242)
    *  [YOLO explained](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006), [ YOLO in python](https://github.com/Garima13a/YOLO-Object-Detection), [YOLO simpler explanation](https://towardsdatascience.com/computer-vision-a-journey-from-cnn-to-mask-r-cnn-and-yolo-part-2-b0b9e67762b1)
    * [YOLOv1 CVPR2016 presentation](https://www.youtube.com/watch?v=NM6lrxy0bxs&feature=youtu.be&list=PLrrmP4uhN47Y-hWs7DVfCmLwUACRigYyT) 
    * [Official slides, resource for YOLO ](https://pjreddie.com/publications/)
    * [YOLOv2 simple explain](http://christopher5106.github.io/object/detectors/2017/08/10/bounding-box-object-detectors-understanding-yolo.html)
  * [v3 2018](https://arxiv.org/pdf/1804.02767.pdf): 
    *  [beginner guide](https://towardsdatascience.com/dive-really-deep-into-yolo-v3-a-beginners-guide-9e3d2666280e),  [theory explained](https://medium.com/analytics-vidhya/yolo-v3-theory-explained-33100f6d193) , [whats new inYOLOv3](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)
    * [YOLOv3 in Keras](https://towardsdatascience.com/object-detection-using-yolov3-using-keras-80bf35e61ce1), [YOLOv3 in PyTorch](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)
  * v4: reading
  * v5: reading
* [SSD](https://arxiv.org/abs/1512.02325)
* [RetinaNet](https://arxiv.org/abs/1708.02002) 2017:  [review on retinanet](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4)
* [EfficientDet: 2019](https://arxiv.org/abs/1911.09070)
* [EfficientNet: 2019](https://arxiv.org/abs/1905.11946)
* MobileNets
* Squeezenet
* ShuffleNet

### Review on Object Detection

* [Object Detection in 20 Years: A Survey](https://arxiv.org/pdf/1905.05055)

![Image from zou2019](https://lh4.googleusercontent.com/XxgASA7WjkiCqsEW-EqXrUaRDNYEyKBkAemKtv4e9rS3AtzVIJBEgysN9in3lpdtLjzXeh5dzkkaYy6DWnWXCvEtjHFOEvaGzxUUrpzpl-NkwcsE32nMWmgBG2uSZFMzBdjWD40Z)

### Advances in 2D Computer Vision

{% embed url="https://towardsdatascience.com/recent-advances-in-modern-computer-vision-56801edab980" %}



### Review: 3D object detection using LiDAR

* \*\*\*\*[**The state of 3D object detection: A review of the state of the art based upon the KITTI leaderboard**](https://towardsdatascience.com/the-state-of-3d-object-detection-f65a385f67a8)\*\*\*\*

![Detector \(LIDAR only\) latency vs vehicle AP](https://miro.medium.com/max/951/1*YtBWthQWmq5bqOytEl51NQ.png)



![Image General approaches for LIDAR+RGB fusion. Images are adapted from MV3D \(Chen et. at. 2016\), F-Pointnet \(Qi et. al. 2017\), ContFuse \(Liang et. al. 2018\), and LaserNet \(Meyer et. al. 2018\).for post](https://miro.medium.com/max/1236/1*N5ilVL6YmjtIHCr-SghsgQ.png)



![Detector \(LIDAR+RGB fusion labeled\) latency vs vehicle APost](https://miro.medium.com/max/936/1*11IfMVEO1yFrI5sz5NAH6A.png)

![Trade-offs between RV and BEV projectionspost](https://miro.medium.com/max/970/1*zYUa1qJsG8Hsp6sh4L9X8w.png)

![Figure from PointPillars](../.gitbook/assets/image%20%2811%29.png)





### 3D Object Orientation Detection

{% embed url="https://towardsdatascience.com/anchors-and-multi-bin-loss-for-multi-modal-target-regression-647ea1974617" %}

{% embed url="https://towardsdatascience.com/orientation-estimation-in-monocular-3d-object-detection-f850ace91411" %}



### Pseudo LiDAR: from 2D to 3D

{% embed url="https://medium.com/swlh/making-a-pseudo-lidar-with-cameras-and-deep-learning-e8f03f939c5f" %}



### Time Series Classification 

* [Deep Learning for Time series classification: a review](https://arxiv.org/pdf/1809.04356.pdf)

* InceptionTime:  [https://towardsdatascience.com/deep-learning-for-time-series-classification-inceptiontime-245703f422db](https://towardsdatascience.com/deep-learning-for-time-series-classification-inceptiontime-245703f422db)

{% embed url="https://arxiv.org/pdf/1809.04356.pdf" %}







