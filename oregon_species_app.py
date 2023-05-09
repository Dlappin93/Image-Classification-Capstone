import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input

class_labels = ['bald_eagle', 'black_bear', 'bobcat', 'canada_lynx', 'columbian_black-tailed_deer', 'cougar',
                'coyote', 'deer', 'elk', 'gray_fox', 'gray_wolf', 'mountain_beaver', 'nutria', 'raccoon', 'raven',
                'red_fox', 'ringtail', 'sea_lions', 'seals', 'virginia_opossum']

# Load the model
model = tf.keras.models.load_model('appmodel.h5')


def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img


def predict_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    predicted_class = class_labels[np.argmax(predictions[0])]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence


def main():

    # Add title and purpose section
    st.title("Oregon Wildlife Classification Using Transfer Learning with Fine-Tuned MobileNetV2 Pretrained Model")

    # Add name and date section
    st.markdown("---")
    st.markdown("Created by: David Lappin")
    st.markdown("Date: May 10, 2023")
    st.markdown("---")

    st.subheader("Project and Application Info:")
    st.markdown("This application performs image classification on 20 different species classes from Oregon. "
                "The predictions are generated using the MobileNetV2 pretrained model that has been fine-tuned to "
                "the wildlife photography data set."
                "\n\n "
                "This application is a final addition to a larger project on image classification using CNN models "
                "in Keras/Tensorflow. The purpose was to try and identify different species so that the same code could "
                "be trained using larger volumes of trail camera footage. In the Future I will be adapting the project"
                "to fine tune the model to trail camera footage for species identification and quantification"
                "The project in its entirety, as well as the code and requirements for this application"
                "can be found on my github: https://github.com/Dlappin93/Image-Classification-Capstone")
    st.markdown("---")
    st.markdown("\n\n")
    st.markdown("\n\n")

    
    st.subheader("Try It Out!:")
    st.markdown("\n\n")
    st.markdown("\n\n")


    # Upload and display the image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Classification button
        if st.button("Classify"):
            predicted_class, confidence = predict_image(uploaded_file)
            st.write("Predicted Class:", predicted_class)
            st.write("Confidence:", f"{confidence:.2f} %")


if __name__ == '__main__':
    main()