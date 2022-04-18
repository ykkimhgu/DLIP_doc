# OpenCV-Python Exercise 





# Beginner Level Exercise



## Exercise 1

Apply Blur filters, Thresholding and Morphology methods on given images for object segmentation. 

[download test image](https://github.com/ykkimhgu/DLIP-src/blob/main/Tutorial_Threshold_Morp/testImage.zip)

![image](https://user-images.githubusercontent.com/38373000/163776140-51398b0d-6cb2-4e02-b21f-6749b1d75049.png)

## Example 2

Choose the appropriate InRange conditions to segment  only ' Blue colored ball'.  Draw the contour and a box over the target object. Repeat for Red, and Yellow balls

[download test image](https://github.com/ykkimhgu/DLIP-src/blob/main/images/color_ball.jpg)

<img src="https://github.com/ykkimhgu/DLIP-src/blob/main/images/color_ball.jpg?raw=true" style="zoom: 33%;" />



## Example 3

Detect Pupil/Iris and draw circles. 



![](C:\Users\ykkim\source\repos\GithubDesktop\DLIP_doc\.gitbook\assets\eyepupil.png)





# Intermediate Level Exercise

## Exercise:  Count number of coins and calculate the total amount

After applying thresholding and morphology, we can identify and extract the target objects from the background by finding the contours around the connected pixels. This technique is used where you need to monitor the number of objects moving on a conveyor belt in an industry process.
Goal: Count the number of the individual coins and calculate the total amount of money.

<img src="https://user-images.githubusercontent.com/38373000/163774968-4415bcc8-418e-49bd-9dcb-c8228f70f405.png" alt="image" style="zoom:50%;" />



**Procedure:**

1. Apply a filter to remove image noises
2. Choose the appropriate threshold value.
3. Apply the appropriate morphology method to segment coins
4. Find the contour and draw the segmented objects.
5. Exclude the contours which are too small or too big
6. Count the number of each different coins(10/50/100/500 won)
7. Calculate the total amount of money.




