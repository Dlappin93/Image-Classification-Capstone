

import streamlit as st
import tensorflow as tf
from PIL import Image


# Define the class names
class_names = ['class1', 'class2', 'class3', ..., 'class20']

# Load the pre-trained Keras model
model = tf.keras.models.load_model('your_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image to match the input size of the model
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255.0  # Normalize pixel values between 0 and 1
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make predictions
def make_prediction(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = class_names[prediction.argmax()]
    return predicted_class

# Streamlit app
st.title("Image Classification")
st.write("Upload an image and the model will predict its class.")

# Image upload
uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    if st.button("Predict"):
        predicted_class = make_prediction(image)
        st.write("Prediction:", predicted_class)