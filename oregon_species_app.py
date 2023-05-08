
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# Streamlit app code
st.title("Image Classifier")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Classify'):
        predicted_class, confidence = predict(image)
        st.write("Predicted Class:", predicted_class)
        st.write("Confidence:", confidence)