# Google Codelab

> > cv2.imshow() is disabled in colab. Instead use
> >
> > from google.colab.patches import cv2\_imshow

## Write a program in Colab

Login your Google Account in Chrome.

Click below or [https://colab.research.google.com/notebooks](https://colab.research.google.com/notebooks/intro.ipynb#recent=true)

{% embed url="https://colab.research.google.com/notebooks/" %}

Start a New Note.

## Loading image file in CoLab

Read the following for detailed information: [Load images by Google Colab](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/load\_data/images.ipynb)

### Method 1: [How to import an image from local drive](https://medium.com/@rk.sarthak01/how-to-import-files-images-in-google-colab-from-your-local-system-46a801b1e568)

```python
# Upload image file from local drive
from google.colab import files
uploaded=files.upload()
```

![](<../../images/image (244).png>)

check the image file

```python
import cv2
from google.colab.patches import cv2_imshow  
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgfName='cat2.jpg'

# Method 1: Open Image  with cv2
img=cv2.imread(imgfName)
cv2_imshow(img)

# Method 2: Open Image  with matplotlib
img=mpimg.imread(imgfName)
plt.imshow(img)
```

### Method 2: How to upload an image from local drive

![](<../../.gitbook/assets/image (240).png>)

## Loading dataset in CoLab
