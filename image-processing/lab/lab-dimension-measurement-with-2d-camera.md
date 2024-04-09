# LAB: Dimension Measurement with 2D camera

## I. Introduction

A company wants to measure the whole dimension of a rectangular object with a smartphone.

You are asked to make an image processing algorithm for an accurate volume measurement of the small object.

![What Software Do I Need with a Leica 3D Imager?](https://tavcotech.com/cdn/shop/articles/what-software-do-i-need-with-a-leica-3d-imager-473023.jpg?v=1702919576) ![147 Ar Tape Images, Stock Photos, 3D objects, & Vectors | Shutterstock](https://www.shutterstock.com/image-photo/ar-mobile-phone-window-measurement-600nw-1989864692.jpg)

### Problem Conditions

* Measure the 3D dimensions (LxWxH) of a small rectangular object
* Assume you know the exact width (**W**) of the target object. You only need to find **L** and **H**.
* The accuracy of the object should be within **3mm**
* You can only use a smartphone (webcam) 2D camera for sensors. No other sensors.
* You cannot know the exact pose of the camera from the object or the table.
* You can use other known dimension objects, such as A4-sized paper, check-board, small square boxes etc
* Try to make the whole measurement process to be as simple, and convenient as possible for the user
  * Using fewer images, using fewer reference objects, etc

## II. Procedure

1. First, understand fully about the design problem.
2. Design the algorithm flow
   * Calibration, Segment the object from the background, Finding corners etc
3. You can use additional reference objects such as A4 paper, known-sized rectangular objects, etc.
   * you will get a higher point if you use the reference object as simple as possible.
4. You must state all the additional assumptions or constraints to make your algorithm work.
   * You are free to add assumptions and constraints such as the reference object can be placed in parallel to the target block etc
   * But, you will get a higher point if you use fewer constraints/assumptions.
5. Use your webcam or smartphone to capture image(s)
   * Use the given experimental setting of the background and 3D object.

![](https://github.com/ykkimhgu/DLIP\_doc/assets/38373000/0174b785-0597-4895-9983-750d4f1fc02b) ![](https://github.com/ykkimhgu/DLIP_doc/assets/38373000/316f7dbe-e2ea-4666-b565-14159da63050)


1. Measure each dimension of the test rectangular object.
   * The exact width (W) of the target object is given.
   * Measure only Height and Length
2. The output image or video should display the measurement numbers.
3. Your algorithm will be validated with other similar test object

## III. Report and Demo Video

#### Lab Report:

* Show what you have done with concise explanations and example results of each necessary process
* In the appendix, show your source code.
* Submit in both PDF and the original file (\*.md etc)

#### Demo Video:

* Create a demo video with a title page showing the course name, data, and your names
* Submit the file on LMS

#### Source Code:

* Zip all the necessary source files.
* Only the source code files. Do not submit image files, project files etc.

####
