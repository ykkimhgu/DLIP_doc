# Masking with Bitwise Operation

## Bitwise Operation



### [◆ ](https://docs.opencv.org/4.x/d2/de8/group\_\_core\_\_array.html#ga60b4d04b251ba5eb1392c34425497e14)bitwise\_and()

| void cv::bitwise\_and | ( | [InputArray](https://docs.opencv.org/4.x/dc/d84/group\_\_core\_\_basic.html#ga353a9de602fe76c709e12074a6f362ba)   | _src1_,                                                                                                                      |
| --------------------- | - | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
|                       |   | [InputArray](https://docs.opencv.org/4.x/dc/d84/group\_\_core\_\_basic.html#ga353a9de602fe76c709e12074a6f362ba)   | _src2_,                                                                                                                      |
|                       |   | [OutputArray](https://docs.opencv.org/4.x/dc/d84/group\_\_core\_\_basic.html#gaad17fda1d0f0d1ee069aebb1df2913c0)  | _dst_,                                                                                                                       |
|                       |   | [InputArray](https://docs.opencv.org/4.x/dc/d84/group\_\_core\_\_basic.html#ga353a9de602fe76c709e12074a6f362ba)   | _mask_ = [`noArray`](https://docs.opencv.org/4.x/dc/d84/group\_\_core\_\_basic.html#gad9287b23bba2fed753b36ef561ae7346)`()`  |
|                       | ) |                                                                                                                   |                                                                                                                              |

{% embed url="https://docs.opencv.org/4.x/d2/de8/group__core__array.html#ga60b4d04b251ba5eb1392c34425497e14" %}





### Example

`bitwise_and(src1,src1,dst1, mask1)`

This means   dst1(I)= src1(I) & src1(I), if mask(I) !=0



src\[10]\[20] = 1111 0001     / / 8-bit

Since  binary logic AND is   **X & X=X**

&#x20;    1111 0001  & 1111 0001 = 1111 0001

&#x20;    dst=src & src  = src



## Masking and Merging



<figure><img src="../../.gitbook/assets/image (224).png" alt=""><figcaption></figcaption></figure>

dst1 and dst2 can be obtained using bitwise operation as

```cpp
bitwise_and(src1, src1, dst1, mask1);
bitwise_and(src2, src2, dst2, mask2);
dst3=dst1+dst2;
```





