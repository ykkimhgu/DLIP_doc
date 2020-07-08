# T\#5: End2End Driving

[Autonomous Driving Cook Book](https://github.com/microsoft/AutonomousDrivingCookbook/tree/master/AirSimE2EDeepLearning)

### Authors:

[**Mitchell Spryn**](https://www.linkedin.com/in/mitchell-spryn-57834545/), Software Engineer II, Microsoft

[**Aditya Sharma**](https://www.linkedin.com/in/adityasharmacmu/), Program Manager, Microsoft

### Modified:

Y.-K Kim, 2020-07-07

## Overview

In this tutorial, you will learn how to train and test an end-to-end deep learning model for autonomous driving using data collected from the [AirSim simulation environment](https://github.com/Microsoft/AirSim). You will train a model to learn how to steer a car through a portion of the Mountain/Landscape map in AirSim using a single front facing webcam for visual input. Such a task is usually considered the "hello world" of autonomous driving, but after finishing this tutorial you will have enough background to start exploring new ideas on your own. Through the length of this tutorial, you will also learn some practical aspects and nuances of working with end-to-end deep learning methods.

Here's a short sample of the model in action:

![car-driving](../../.gitbook/assets/car_driving.gif)

#### Demo

[https://youtu.be/CauKo089zm0](https://youtu.be/CauKo089zm0)

#### How to embed 

[https://youtu.be/CauKo089zm0](https://www.youtube.com/watch?v=cFtnflNe5fM)

## Structure of this tutorial

The code presented in this tutorial is written in [Keras](https://keras.io/), a high-level deep learning Python API capable of running on top of [CNTK](https://www.microsoft.com/en-us/cognitive-toolkit/), [TensorFlow](https://www.tensorflow.org/) or [Theano](http://deeplearning.net/software/theano/index.html). The fact that Keras lets you work with the deep learning framework of your choice, along with its simplicity of use, makes it an ideal choice for beginners, eliminating the learning curve that comes with most popular frameworks.

This tutorial is presented to you in the form of Python notebooks. Python notebooks make it easy for you to read instructions and explanations, and write and run code in the same file, all with the comfort of working in your browser window. You will go through the following notebooks in order:

**1.** [**DataExplorationAndPreparation**](https://github.com/microsoft/AutonomousDrivingCookbook/blob/master/AirSimE2EDeepLearning/DataExplorationAndPreparation.ipynb)

[Modified notebook](https://github.com/ykkimhgu/gitbook_docs/tree/6fa008a242b585a87e4d46af33d9f42ff73f43ae/airsim/docs/tutorial/DataExplorationAndPreparation.ipynb)

**2.** [**TrainModel**](https://github.com/microsoft/AutonomousDrivingCookbook/blob/master/AirSimE2EDeepLearning/TrainModel.ipynb)

**3.** [**TestModel**](https://github.com/microsoft/AutonomousDrivingCookbook/blob/master/AirSimE2EDeepLearning/TestModel.ipynb)

If you have never worked with Python notebooks before, we highly recommend [checking out the documentation](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html).

## Prerequisites and setup

### Background needed

You should be familiar with the basics of neural networks and deep learning. You are not required to know advanced concepts like LSTMs or Reinforcement Learning but you should know how Convolutional Neural Networks work. A really good starting point to get a strong background in a short amount of time is [this highly recommended book on the topic](http://neuralnetworksanddeeplearning.com/) written by Michael Nielsen. It is free, very short and available online. It can provide you a solid foundation in less than a week's time.

You should also be comfortable with Python. At the very least, you should be able to read and understand code written in Python.

### Environment Setup

1. Clone the tutorial git repository '[https://github.com/microsoft/AutonomousDrivingCookbook](https://github.com/microsoft/AutonomousDrivingCookbook)'
   * Download the repository as zip file or
   * From a terminal window, change to the local directory where you want to clone and run

     ```text
        git clone https://github.com/microsoft/AutonomousDrivingCookbook.git
     ```

     > _**Note**_ [Read how to clone a repository](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) for more help
2. [Install AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy). Be sure to add the location for the AzCopy executable to your system path.
3. [Install Anaconda](https://conda.io/docs/user-guide/install/index.html) with Python 3.5 or higher.

> _**Note**_: Use virtual environment to install Tensorflow, Keras and other dependencies _**Note**_ [Read how to use virtual environment in Anaconda](tutorial_end_to_end_driving.md)

1. [Install Tensorflow](https://www.tensorflow.org/install/install_windows)

   example: if the virtual environment name is 'tf22'

   ```text
        activate tf22
        conda install -c anaconda tensorflow-gpu
   ```

> _**Note**_ [Read how to install Tensorflow with Anaconda](tutorial_end_to_end_driving.md)

1. [Install h5py](http://docs.h5py.org/en/latest/build.html)

   ```text
    `conda install h5py`
   ```

2. [Install Keras](https://keras.io/#installation) and [configure the Keras backend](https://keras.io/backend/) to work with TensorFlow \(default\).

   `conda install -c anaconda keras`

3. Install the other dependencies. From your anaconda **virtual** environment, run [InstallPackages.py](https://github.com/microsoft/AutonomousDrivingCookbook/blob/master/AirSimE2EDeepLearning/InstallPackages.py) as root or administrator. This installs the following packages into your environment:
   * jupyter
   * matplotlib v. 2.1.2
   * image
   * keras\_tqdm
   * opencv
   * msgpack-rpc-python
   * pandas
   * numpy
   * scipy

### Simulator Package

We have created a standalone build of the AirSim simulation environment for the tutorials in this cookbook. [You can download the build package from here](https://airsimtutorialdataset.blob.core.windows.net/e2edl/AD_Cookbook_AirSim.7z). Consider using [AzCopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy), as the file size is large. After downloading the package, unzip it and run the PowerShell command

`.\AD_Cookbook_Start_AirSim.ps1 landscape` ![lanscape](../../.gitbook/assets/tutorial_ADcook1.png)

to start the simulator in the landscape environment.

### Hardware

It is highly recommended that a GPU is available for running the code in this tutorial. While it is possible to train the model using just a CPU, it will take a very long time to complete training. This tutorial was developed with an Nvidia GTX970 GPU, which resulted in a training time of ~45 minutes.

If you do not have a GPU available, you can spin up a [Deep Learning VM on Azure](https://azuremarketplace.microsoft.com/en-us/marketplace/apps/microsoft-ads.dsvm-deep-learning), which comes with all the dependencies and libraries installed \(use the provided py35 environment if you are using this VM\).

### Dataset

The dataset for the model is quite large. [You can download it from here](https://aka.ms/AirSimTutorialDataset). The first notebook will provide guidance on how to access the data once you have downloaded it. The final uncompressed data set size is approximately 3.25GB \(which although is nothing compared to the petabytes of data needed to train an actual self-driving car, should be enough for the purpose of this tutorial\).

![dataset](../../.gitbook/assets/tutorial_ADcook_dataset.png)

### A note from the curators

We have made our best effort to ensure this tutorial can help you get started with the basics of autonomous driving and get you to the point where you can start exploring new ideas independently. We would love to hear your feedback on how we can improve and evolve this tutorial. We would also love to know what other tutorials we can provide you that will help you advance your career goals. Please feel free to use the GitHub issues section for all feedback. All feedback will be monitored closely. If you have ideas you would like to [collaborate](https://github.com/ykkimhgu/gitbook_docs/tree/6fa008a242b585a87e4d46af33d9f42ff73f43ae/airsim/docs/README.md#contributing) on, please feel free to reach out to us and we will be happy to work with you.

## ISSUES

### 1. Fails to create test.h5 and eval.h5 in DataExplorationAndPreparation

#### Problem description

Stops with error in generating test/validation dataset of .h5 files in **DataExplorationAndPreparation** It seems it creates **train.h5** but does not create **test.h5** and **eval.h5**

#### Problem details

When running this code in **DataExplorationAndPreparation**

```text
    train_eval_test_split = [0.7, 0.2, 0.1]
    full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]
    Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split)
```

Iteration stops with this output message. It seems it creates **train.h5** but does not create **test.h5** and **eval.h5** Codes before this line worked fine.

```text
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_1...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_2...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_3...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_4...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_5...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/normal_6...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/swerve_1...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/swerve_2...
    Reading data from ../dataset/EndToEndLearningRawData/data_raw/swerve_3...
    Processing ../dataset/data_cooked/train.h5...
    ---------------------------------------------------------------------------
    StopIteration                             Traceback (most recent call last)
    ~\Git_Program\AirSim\ADCookBook\Tutorial_E2EDeepLearning\Cooking.py in generatorForH5py(data_mappings, chunk_size)
        129                 raise StopIteration
    --> 130     raise StopIteration
        131 

    StopIteration: 

    The above exception was the direct cause of the following exception:

    RuntimeError                              Traceback (most recent call last)
    <ipython-input-13-bce568193587> in <module>
        1 train_eval_test_split = [0.7, 0.2, 0.1]
        2 full_path_raw_folders = [os.path.join(RAW_DATA_DIR, f) for f in DATA_FOLDERS]
    ----> 3 Cooking.cook(full_path_raw_folders, COOKED_DATA_DIR, train_eval_test_split)

    ~\Git_Program\AirSim\ADCookBook\Tutorial_E2EDeepLearning\Cooking.py in cook(folders, output_directory, train_eval_test_split)
        192     for i in range(0, len(split_mappings), 1):
        193         print('Processing {0}...'.format(output_files[i]))
    --> 194         saveH5pyData(split_mappings[i], output_files[i])
        195         print('Finished saving {0}.'.format(output_files[i]))
        196 

    ~\Git_Program\AirSim\ADCookBook\Tutorial_E2EDeepLearning\Cooking.py in saveH5pyData(data_mappings, target_file_path)
        162         dset_previous_state[:] = previous_state_chunk
        163 
    --> 164         for image_names_chunk, label_chunk, previous_state_chunk in gen:
        165             image_chunk = np.asarray(readImagesFromPath(image_names_chunk))
        166 
    RuntimeError: generator raised StopIteration
```

#### Experiment/Environment details

* Tutorial used: _AirSimE2EDeepLearning_-&gt; _DataExplorationAndPreparation_
* Environment used: _landscape_
* Versions of artifacts used \(if applicable\): _Python 3.7.7 Keras 2.3.1 Using Tensorflow backened_, _h5py_

