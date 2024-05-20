# Handwritten Digit Recognition using Machine Learning and Deep Learning with the MNIST Dataset
## Overview
This project focuses on developing a machine learning model to recognize handwritten digits using the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of handwritten digits ranging from 0 to 9. Out of these, 60,000 images are used for training and 10,000 for testing. The goal is to leverage Python and libraries like Keras to build a Convolutional Neural Network (CNN) that can accurately classify handwritten digits.

## Technical Steps:
```1. Loading and Splitting Data:``` The MNIST dataset is loaded and split into training and testing sets. The training set consists of 60,000 images and labels, while the testing set consists of 10,000 images and labels.

```2. Data Preprocessing:```
The images are reshaped to fit the neural network's input requirements. Additionally, the digit labels are converted to a one-hot encoded format to improve the model's performance in classifying digits.

```3. Building and Training the CNN:```
A Convolutional Neural Network (CNN) is designed and trained using the Keras library. The model consists of convolutional layers for feature extraction and dense layers for classification. The model is trained on the training dataset and validated on the testing dataset.

```4. Predicting Handwritten Digits:```
An external image is loaded and preprocessed to match the format of the training data. The trained model is then used to predict the digit in the image, and the result is saved to a text file (output.txt).



## Setup and Installation
1. Clone this repository to your local machine.
2. Navigate to the project directory.
3. Install the necessary libraries using pip:
```
pip install keras numpy pillow
```
