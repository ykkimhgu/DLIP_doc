# BottleNeck Unit

To reduce complexity while maintaining minimum accuracy loss. Several different types of bottleneck units are

1. Convolution 3x3
2. Inception Bottleneck
3. ResNet Bottleneck Unit
4. Mobile v2 bottleneck
5. Shuffle Bottleneck

![](https://github.com/ykkimhgu/DLIP_doc/tree/3298e5d2a4b6369e5cef7973dd93eef44ca7addf/.gitbook/assets/image%20%28194%29.png)

## Multiplication cost

Assume Input: WxHxC and Output : WxHxC.

Let m=C/4. g=2, C=16

1. Convolution 3x3:  9CCHW  =  2304HW
2. ResNet Bottleneck: HW\(2mC+9mm\)=  \(17/16\)CCHW= 272HW
3. MobileNet Bottleneck: HW\(2mC+9m\) =  \(\(2/4\)CC+9C/4\)HW = 164HW
4. ShuffleNet Bottleneck: HW\(2mC/g+9m\) = \(\(2/4g\)CC+9C/4\)HW=100HW

