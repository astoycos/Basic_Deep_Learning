# Mini_Project2: Basic Neural Network Design for Pneumonia diagnosis   


This program was created to try and classify DICOM chest X-Rays as either Pneumonia positive or negative based on lung opacitites, 
Specifically It builds a simple neural network to try and accomplish the task. The data was taken from https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data. 


## Prerequisites

Required Packages 

```
Python v3.6
Tensorflow v1.5.0
keras 
shuntil 
csv
pydicom 
numpy 
pandas
skimage
itertools
matplotlib
pickle

```

## Program Summary 

### 1. pneumonia_posneg.py

The first module exists to preprocess the data and create the training, validation, and testing sets.  The RSNA images are saved in DICOM
medical format and the labels are saves in a .csv file. The data is stored in four directories, stage_1_test_images.zip stage_1_train_images.zip stage_1_train_labels.csv.zip. The first program 
reads through these directories and begins by creating two dictionaries, on for the testing and validation sets. The dictionaries store a key with the image filename and 
then encodes either a 0(pneumonia negative) or 1(pneumonia_Positive) for the value.  Then it loops throught these dictionaries and converts all the .dcm files to .png files in order to 
work with the keras flow_from_directory function. specifically the directories much contain the following hierarchy. 
```
data/
    train/
        pneumonia/
            0001.jpg
            0002.jpg
            ...
        no_pneumonia/
            0001.jpg
            0002.jpg
            ...
    validation/
        pneumonia/
            0001.jpg
            0002.jpg
            ...
        no_pneumonia/
            0001.jpg
            0002.jpg
            ...
```
Lastly it creates a test directory and loads all of the test images into it for further possible model verification and testing

### 2. pneumonia_posneg_model.py

The Second module heavily depends on Keras a python wrapper for tensorflow. It first creates the neural network loosely following Lenet5, an existing 
image identification software. Specifically the network includes three convolution and Max Pooling layers are follwed by three dense layers all of which are activated by the Relu function. 
The network concludes with a sigmoid activation to narrow the output down to a single one hot vector signaling either pnemonia positive or negative
It also saves the history of the model to the current working directory for use by the next module 

### 3. pneumonia_posneg_eval.py

The last module simply takes the history dictionary returned by the Keras fit_model() function and creates a subplot of the Train/validation accuracy
and loss functions for each epoch. 


## Execution 

Once all the required packages have been installed you are ready to run the program 
1. Unzip the data files with terminal and leave in current directory 

```
$ Unzip stage_1_test_images.zip
$ Unzip	stage_1_train_images.zip
$ Unzip	stage_1_train_labels.csv.zip
```
2. execute the modules in the following order 

```
$ Python3 pneumonia_posneg.py
$ Python3 pneumonia_posneg_model.py
$ Python3 pneumonia_posneg_eval.py
```

## A Short Neural Net Architecture Comarison: LeNet5 and Resnet 
LeNet was released in 1988 by Yann LeCun and is a pioneering architecture that paved the way for many of the modern deep learning architectutres used today. It was revolutionary due to the fact that never before had there been a network which used convolutions to extract various image features.  Also, it was built during an era where hardware was a major constrant so being able to tune the various convolutional layers made it efficient enough to run in the pre GPU era.  Specifically the platform was built using three major blocks. First the image is convoluted by various sized filters to extract features, as you go deeper in the network the feature maps change from simply reporesening lines and edges, to being able to recognize macro objects.  Pooling layers follow each convolution and serve to extract the most significant data within a feature map while also decreasing the size of the layer.  Lastly a non-linear activation function is applied, such as a tanh or sigmoid equation. Lastly is a set of dense fully connected layers to serve as a final classifier. Due to the simplistic and tunable nature of this architecture I decided to model the basic pneumonia network following many of the same guidelines.   


## Authors

* **Andrew Stoycos** 

## Acknowledgments

* Specific code sourses can be found in module source code 
* Genereal code schematic provided by https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
* Architecture Comparison sources 
**https://towardsdatascience.com/neural-network-architectures-156e5bad51ba


