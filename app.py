import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import base64

# Load the pre-trained model
model = load_model('mnist_cnn_model.h5')

# Streamlit interface
st.set_page_config(page_title="Handwritten Digit Recognition", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for styling
st.markdown("""
<style>
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #d3d3d3;
        text-align: center;
        margin-bottom: 30px;
    }
    .file-upload {
        font-size: 16px;
        font-weight: bold;
        color: #d3d3d3;
        text-align: center;
        margin-top: 30px;
    }
    .result {
        font-size: 24px;
        font-weight: bold;
        color: #d3d3d3;
        text-align: center;
        margin-top: 30px;
    }
    .download-button {
        display: inline-block;
        padding: 10px 20px;
        background-color: #d3d3d3;
        color: #ffffff;
        font-size: 16px;
        font-weight: bold;
        text-decoration: none;
        border-radius: 5px;
        transition: background-color 0.3s ease;
    }
    .download-button:hover {
        background-color: #555555;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">Handwritten Digit Recognition</div>', unsafe_allow_html=True)
st.markdown("Upload an image of a handwritten digit (JPEG or PNG format) and let the model predict the digit.")

# File uploader
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    # Open the image
    image = Image.open(uploaded_file).convert('L')
    image = image.resize((28, 28))

    # Display the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = image_array.reshape((1, 28, 28, 1))

    # Predict the digit
    prediction = model.predict(image_array)
    predicted_label = np.argmax(prediction)

    # Display the predicted digit
    st.markdown(f'<div class="result">Predicted Digit: <span style="font-size: 48px;">{predicted_label}</span></div>', unsafe_allow_html=True)

    # Create a downloadable text file with the predicted digit
    text_file = f"Predicted Digit: {predicted_label}"
    b64 = base64.b64encode(text_file.encode()).decode()
    href = f'<a class="download-button" href="data:file/txt;base64,{b64}" download="predicted_digit.txt">Download Result</a>'
    st.markdown(href, unsafe_allow_html=True)

else:
    st.markdown('<div class="file-upload">Please upload an image file.</div>', unsafe_allow_html=True)