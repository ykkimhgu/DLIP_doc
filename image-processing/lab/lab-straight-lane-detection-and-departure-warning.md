# LAB: Straight Lane Detection and Departure Warning

## Introduction

In this project, you are required to create a program that displays lane departure warning message from the road lane detection.

To simplify this lab, only the straight lanes are to be detected and other difficult driving environment such as curved lanes and road intersections are ignored.

You can use any algorithm you learnt in class. If you want to use any other algorithms which are not covered in the lecture, you should describe the principle of those algorithms fully in the report. The evaluation of the project will be based on the success detection rate of the lanes of the test and validation video clips.

You must explain the whole process with an algorithm flowchart. Show and explain selected example results for each image process.

Demo Video

{% embed url="https://youtu.be/onMBtLw2ag8" %}

## Methods/Requirement

* Download test videos
  * [Test Video- Straigt Line only](https://drive.google.com/file/d/13ZiuL3sDKcptWPdPC\_KzsUNcWIStWpWl/view?usp=sharing)
  * [Test Video - Lane Change](https://drive.google.com/file/d/1eO\_fmHNfhX0FEU1rtiVpZxfOJEoksSLf/view?usp=sharing)
* Detect and segment lanes only from the test image frames.
* Draw only one line per lane
* Display an approximated rate of the lane departure (unit:%)
* If the vehicle has departed from the current lanes then show a warning sign.
* Display FPS (frame per sec) for each frame processing

## Score

Your program will be evaluated on the validation dataset which will not be provided. The lab score will be based on the algorithm performance, algorithm design process, and the quality of the report.

Also, your code will be tested on another test video file.

Do not duplicate the program from other website or from previous lab report. You can refer to any technical reports as long as the proper citation is used.

### Report \[50%]

The report must include

* Introduction
* Algorithm
  * Overview: flowchart/block diagram  (No Hand drawing)
  * Each Image Process steps and results
* Final Result & Analysis
* Conclusion
* Appendix: program code

### Demo Video \[50%]

* Will be scored depending on the detection accuracy and speed
* Your code will be tested on other similar test video.

## Submit

* Report in both PDF and .md source code with text image files.
* Demo movie clip links included in the report
* Source files

## Appendix

### Some Examples:

![](<../../.gitbook/assets/image (95).png>)

![](<../../.gitbook/assets/image (94).png>)
