# Masking with Bitwise Operation

## Bitwise Operation

`bitwise_and(), bitwise_not()` and more



Computes bitwise conjunction of the two arrays (src1,  src2) and calculates the **per-element bit-wise** conjunction of two arrays or an array and a scalar.



### bitwise\_and()

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



Assume   src\[10]\[20] = 1111 0001     / / 8-bit data per  pixel


### Case 1
Since  binary logic AND is   **X & X=X**

&#x20;    1111 0001  & 1111 0001 = 1111 0001

The output of bitwise operation becomes

&#x20;     dst=src & src  = src


### Case 2
Since  binary logic AND is   **X & 1=X**

&#x20;    1111 0001  & 1111 1111 = 1111 0001

The output of bitwise operation becomes

&#x20;     dst=src & 1  = src

### Case 3
Since  binary logic AND is   **X & 0=0**

&#x20;    1111 0001  & 0000 0000 = 0000 0000

The output of bitwise operation becomes all black.



## Masking and Merging



<figure><img src="../../.gitbook/assets/image (224).png" alt=""><figcaption></figcaption></figure>

dst1 and dst2 can be obtained using bitwise operation as

For both  1-CH and 3-CH images

```cpp
bitwise_and(src1, src1, dst1, mask1);
bitwise_and(src2, src2, dst2, mask2);
dst3=dst1+dst2;
```

Also, you can apply as

```cpp
dst1=src1 & mask1;         // This is NOT the same as  && operation 
dst2=src2 & mask2;
dst3=dst1 + dst2;
```



