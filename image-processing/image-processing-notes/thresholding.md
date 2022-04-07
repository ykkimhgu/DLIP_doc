# Thresholding

$$
a = b *b
$$

$${\label{eq.1}} F=\alpha A+\beta W=\beta(\frac{\alpha}{\beta}A+W)$$

\begin{equation}{\label{eq.1\}} F=\alpha A+\beta W=\beta(\frac{\alpha}{\beta}A+W) \end{equation},

**A Short Summary of Thresholding Algorithm**

* The basic concept of thresholding: to segment objects from the background based on intensity values.
* A simple method is making the output result as a binary image as

![](../../.gitbook/assets/0.png)

* Multiple thresholding

![](<../../.gitbook/assets/1 (1).png>)

1. Analyze the intensity histogram and select the initial estimation of ![](<../../.gitbook/assets/4 (1) (1) (1) (1) (1).png>)(usually the mean of the image intensity). Let the intensity of the input image is defined as g(x,y).
2. Segment the image by two groups on the histogram using the value of ![](../../.gitbook/assets/6.png)

![](<../../.gitbook/assets/7 (1).png>)

1. Find the mean of ![](../../.gitbook/assets/8.png)and ![](../../.gitbook/assets/9.png)(i.e. _m1_ and _m2_)
2. The new ![](<../../.gitbook/assets/10 (1).png>)value at kth iteration

![](../../.gitbook/assets/11.png)

1. repeat from step 2 until ![](<../../.gitbook/assets/12 (1).png>), where ![](../../.gitbook/assets/13.png)
2. OTSU’s method.
3. The aim is to maximize the between-class variance based on the histogram of an image
4. First, calculate the normalized histogram ![](<../../.gitbook/assets/14 (1).png>), with _ni_ is the number of pixels with the intensity level _I_, and it should satisfy

![](<../../.gitbook/assets/15 (1).png>)

* Let us define the mean intensity of the entire image as

![](../../.gitbook/assets/16.png)

which is equivalent to

![](<../../.gitbook/assets/40 (1) (1) (1) (1) (1).png>)

![](<../../.gitbook/assets/20 (1).png>)

Probability of![](<../../.gitbook/assets/21 (1).png>), given that ![](<../../.gitbook/assets/22 (1).png>)comes from the class ![](<../../.gitbook/assets/23 (1).png>)

![](<../../.gitbook/assets/24 (1).png>) (using Bayes’ formula)

Note: Bayes formula

![](<../../.gitbook/assets/25 (1).png>)

* \*
* Then, the mean of intensity of class ![](../../.gitbook/assets/30.png) becomes

![](<../../.gitbook/assets/31 (2) (1) (1).png>)

* Similarly, the mean of intensity of class ![](../../.gitbook/assets/32.png) becomes

![](<../../.gitbook/assets/33 (1).png>)

where ![](<../../.gitbook/assets/34 (1).png>) and ![](../../.gitbook/assets/35.png)

* The cumulative mean intensity from ‘0’ up to level ![](<../../.gitbook/assets/36 (1).png>)is defined as

![](../../.gitbook/assets/37.png) // ![](<../../.gitbook/assets/31 (2) (1) (1) (1) (1).png>)

* Thus, we can express the total mean intensity as

![](../../.gitbook/assets/39.png)

since the total mean intensity is ![](<../../.gitbook/assets/40 (1) (1) (1) (1).png>)

* To evaluate the ‘goodness’ of the threshold values of ![](../../.gitbook/assets/41.png), we can design a score

![](../../.gitbook/assets/42.png)

![](<../../.gitbook/assets/43 (1).png>)is the global variance

![](<../../.gitbook/assets/44 (1).png>)

![](<../../.gitbook/assets/45 (2) (1) (1) (1).png>)is the between-class variance

![](<../../.gitbook/assets/46 (1).png>)

The further the two means of ![](../../.gitbook/assets/47.png)and ![](../../.gitbook/assets/48.png)are from each other, the larger ![](<../../.gitbook/assets/45 (2) (1) (1).png>) will be

larger value of _η._

To make the calculation simpler, we transform the formula as

![](<../../.gitbook/assets/50 (1).png>)

The Procedure of Otsu Method

![](../../.gitbook/assets/51.png)

Aim: obtain the maximum ![](<../../.gitbook/assets/62 (1) (2) (1) (1) (1) (2).png>) from the calculation of ![](<../../.gitbook/assets/62 (1) (2) (1) (1) (1) (3).png>)for all values of _k_

1. Apply an image filter prior to thresholding.
2. Compute the normalized histogram ![](../../.gitbook/assets/54.png)
3. Compute the cumulative sum ![](<../../.gitbook/assets/55 (1).png>), ![](<../../.gitbook/assets/56 (1) (1) (1) (1).png>)to![](<../../.gitbook/assets/57 (1) (1) (1) (1) (1) (1).png>)
4. Compute the cumulative mean ![](<../../.gitbook/assets/58 (1).png>), ![](<../../.gitbook/assets/56 (1) (1) (1) (1) (1) (1).png>)to![](<../../.gitbook/assets/57 (1) (1) (1) (1).png>)
5. Compute the global intensity mean ![](<../../.gitbook/assets/61 (1).png>)
6. Compute ![](<../../.gitbook/assets/62 (1) (2) (1) (1) (2).png>), for all ![](<../../.gitbook/assets/63 (1).png>)
7. Find k\* at which ![](<../../.gitbook/assets/64 (1).png>) is at maximum
8. Apply threshold at ![](<../../.gitbook/assets/65 (1).png>)\*
9. Local thresholding

Method 1. Image partitioning

* Subdivide an image into non overlapping rectangles. Apply otsu threshold in each sub division.
* Works well when the objects and background occupy reasonably comparable size.
* But fails if either object or background is small.

Method 2. Based on local image property

![](<../../.gitbook/assets/70 (1).png>)

![](../../.gitbook/assets/71.png)

preferable if background is nearly uniform.

Method 3. Moving average.

* Scan line by line in zigzag to reduce illumination effect

![](../../.gitbook/assets/72.png)

Where ![](<../../.gitbook/assets/73 (1).png>)is intensity of the point at step ![](<../../.gitbook/assets/74 (1).png>)in the number of points area in M.A ![](../../.gitbook/assets/75.png)

Use ![](../../.gitbook/assets/76.png)

See example with text image corrupted by spot shading

![](../../.gitbook/assets/77.png) ![](../../.gitbook/assets/78.png)
