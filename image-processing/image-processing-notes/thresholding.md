# Thresholding

$$
a = b *b
$$

$${\label{eq.1}} F=\alpha A+\beta W=\beta(\frac{\alpha}{\beta}A+W)$$

\begin{equation}{\label{eq.1}} F=\alpha A+\beta W=\beta\(\frac{\alpha}{\beta}A+W\) \end{equation},

**A Short Summary of Thresholding Algorithm**

* The basic concept of thresholding: to segment objects from the background based on intensity values.
* A simple method is making the output result as a binary image as

![](../../.gitbook/assets/0.png)

* Multiple thresholding

![](../../.gitbook/assets/1%20%281%29.png)

1. Analyze the intensity histogram and select the initial estimation of ![](../../.gitbook/assets/5%20%281%29.png)\(usually the mean of the image intensity\). Let the intensity of the input image is defined as g\(x,y\).
2. Segment the image by two groups on the histogram using the value of ![](../../.gitbook/assets/6.png)

![](../../.gitbook/assets/7%20%281%29.png)

1. Find the mean of ![](../../.gitbook/assets/8.png)and ![](../../.gitbook/assets/9.png)\(i.e. _m1_ and _m2_\)
2. The new ![](../../.gitbook/assets/10%20%281%29.png)value at kth iteration

![](../../.gitbook/assets/11.png)

1. repeat from step 2 until ![](../../.gitbook/assets/12%20%281%29.png), where ![](../../.gitbook/assets/13.png)
2. OTSU’s method.
3. The aim is to maximize the between-class variance based on the histogram of an image
4. First, calculate the normalized histogram ![](../../.gitbook/assets/14%20%281%29.png), with _ni_ is the number of pixels with the intensity level _I_, and it should satisfy

![](../../.gitbook/assets/15%20%281%29.png)

* Let us define the mean intensity of the entire image as

![](../../.gitbook/assets/16.png)

which is equivalent to

![](../../.gitbook/assets/17.png)

![](../../.gitbook/assets/20%20%281%29.png)

Probability of![](../../.gitbook/assets/21%20%281%29.png), given that ![](../../.gitbook/assets/22%20%281%29.png)comes from the class ![](../../.gitbook/assets/23%20%281%29.png)

![](../../.gitbook/assets/24%20%281%29.png) \(using Bayes’ formula\)

Note: Bayes formula

![](../../.gitbook/assets/25%20%281%29.png)

* \* 
* Then, the mean of intensity of class ![](../../.gitbook/assets/30.png) becomes

![](../../.gitbook/assets/31.png)

* Similarly, the mean of intensity of class ![](../../.gitbook/assets/32.png) becomes

![](../../.gitbook/assets/33%20%281%29.png)

where ![](../../.gitbook/assets/34%20%281%29.png) and ![](../../.gitbook/assets/35.png)

* The cumulative mean intensity from ‘0’ up to level ![](../../.gitbook/assets/36%20%281%29.png)is defined as

![](../../.gitbook/assets/37.png) // ![](../../.gitbook/assets/38%20%281%29.png)

* Thus, we can express the total mean intensity as

![](../../.gitbook/assets/39.png)

since the total mean intensity is ![](../../.gitbook/assets/40%20%281%29.png)

* To evaluate the ‘goodness’ of the threshold values of ![](../../.gitbook/assets/41.png), we can design a score

![](../../.gitbook/assets/42.png)

![](../../.gitbook/assets/43%20%281%29.png)is the global variance

![](../../.gitbook/assets/44%20%281%29.png)

![](../../.gitbook/assets/45.png)is the between-class variance

![](../../.gitbook/assets/46%20%281%29.png)

The further the two means of ![](../../.gitbook/assets/47.png)and ![](../../.gitbook/assets/48.png)are from each other, the larger ![](../../.gitbook/assets/49.png) will be

larger value of _η._

To make the calculation simpler, we transform the formula as

![](../../.gitbook/assets/50%20%281%29.png)

The Procedure of Otsu Method

![](../../.gitbook/assets/51.png)

Aim: obtain the maximum ![](../../.gitbook/assets/52%20%281%29.png) from the calculation of ![](../../.gitbook/assets/53%20%281%29.png)for all values of _k_

1. Apply an image filter prior to thresholding.
2. Compute the normalized histogram ![](../../.gitbook/assets/54.png)
3. Compute the cumulative sum ![](../../.gitbook/assets/55%20%281%29.png), ![](../../.gitbook/assets/56%20%281%29.png)to![](../../.gitbook/assets/57%20%281%29.png)
4. Compute the cumulative mean ![](../../.gitbook/assets/58%20%281%29.png), ![](../../.gitbook/assets/59%20%281%29.png)to![](../../.gitbook/assets/60.png)
5. Compute the global intensity mean ![](../../.gitbook/assets/61%20%281%29.png)
6. Compute ![](../../.gitbook/assets/62%20%281%29.png), for all ![](../../.gitbook/assets/63%20%281%29.png)
7. Find k\* at which ![](../../.gitbook/assets/64%20%281%29.png) is at maximum
8. Apply threshold at ![](../../.gitbook/assets/65%20%281%29.png)\*
9. Local thresholding

Method 1. Image partitioning

* Subdivide an image into non overlapping rectangles. Apply otsu threshold in each sub division.
* Works well when the objects and background occupy reasonably comparable size.
* But fails if either object or background is small.

Method 2. Based on local image property

![](../../.gitbook/assets/70%20%281%29.png)

![](../../.gitbook/assets/71.png)

preferable if background is nearly uniform.

Method 3. Moving average.

* Scan line by line in zigzag to reduce illumination effect

![](../../.gitbook/assets/72.png)

Where ![](../../.gitbook/assets/73%20%281%29.png)is intensity of the point at step ![](../../.gitbook/assets/74%20%281%29.png)in the number of points area in M.A ![](../../.gitbook/assets/75.png)

Use ![](../../.gitbook/assets/76.png)

See example with text image corrupted by spot shading

![](../../.gitbook/assets/77.png)![](../../.gitbook/assets/78.png)

