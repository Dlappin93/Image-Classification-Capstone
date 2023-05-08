
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests


# Load the pre-trained model
model = tf.keras.models.load_model('appmodel.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class names
class_names = ['bald_eagle', 'black_bear', 'bobcat', 'canada_lynx', 'columbian_black-tailed_deer', 'cougar',
               'coyote', 'deer', 'elk', 'gray_fox', 'gray_wolf', 'mountain_beaver', 'nutria', 'raccoon', 'raven',
               'red_fox', 'ringtail', 'sea_lions', 'seals', 'virginia_opossum']

# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((224, 224))  # Resize the image to match the input size of the model
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    return class_names[predicted_class], confidence

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