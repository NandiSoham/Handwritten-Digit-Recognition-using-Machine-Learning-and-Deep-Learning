# Handwritten Digit Recognition

This is a machine learning based web application built using Python and Streamlit that allows users to upload images of handwritten digits and have them recognized by a pre-trained deep learning model. The model is trained on the MNIST dataset, which consists of 70,000 images of handwritten digits (60,000 for training and 10,000 for testing), with each image being 28x28 pixels in grayscale. The project includes two main components: training the CNN model and deploying the web application.

![handwritten](https://github.com/NandiSoham/Handwritten-Digit-Recognition-using-Machine-Learning-and-Deep-Learning/assets/56528719/9eb92dd5-b7d2-40dc-a27f-e56d9f2fa2d9)


## Live Link
You can access the live web application [here](https://handwrittendigitrecognition.streamlit.app/).

## Project Structure

The project consists of the following files:

- `app.py`: The main Streamlit application file that handles the user interface, file upload, image preprocessing, and digit prediction.
- `train_model.py`: A Python script for training the Convolutional Neural Network (CNN) model and saving it to a file (mnist_cnn_model.h5).
- `handwritten_digit_recognition.ipynb`: A Jupyter Notebook containing the code for experimenting with loading the MNIST dataset, data preprocessing, model building, training, and evaluation.
- `requirements.txt`: A text file listing the required Python packages and their versions.

## Requirements
- keras
- numpy
- tensorflow
- streamlit

We can install these packages using the `requirements.txt` file


## Training the Model
The train_model.py script performs the following tasks:

1. Data Loading and Preprocessing: Loads the MNIST dataset and preprocesses it for training.
2. Data Augmentation: Applies random transformations to the training data to improve the model's robustness.
3. Model Building: Constructs a CNN model using Keras 
4. Model Training: Trains the model using the Adam optimizer, with early stopping and learning rate reduction callbacks.
5. Model Saving: Saves the trained model to mnist_cnn_model.h5.

We need to train the model from scratch and run the `train_model.py` script. This script will train the CNN on the MNIST dataset and save the model to `mnist_cnn_model.h5`.
```
python train_model.py

```

## Running the Web Application

The app.py script sets up a Streamlit web application with the following features:
1. File Uploader: Allows users to upload an image file (JPEG or PNG format).
2. Image Display: Shows the uploaded image.
3. Image Preprocessing: Converts the image to grayscale, resizes it to 28x28 pixels, and normalizes the pixel values.
4. Prediction: Uses the pre-trained model to predict the digit in the uploaded image.
5. Result Display: Shows the predicted digit.
6. Download Option: Provides a link to download the predicted result as a text file.

To run the Streamlit web application, execute the app.py script:
```
streamlit run app.py

```

## Example Workflow
1. Upload an Image: Click on the file uploader to select an image of a handwritten digit.
2. View the Image: The uploaded image will be displayed on the screen.
3. Get Prediction: The model will predict the digit and display the result.
4. Download Result: Click the download link to save the prediction to a text file.
