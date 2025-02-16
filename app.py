import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pickle

# Load the full model using pickle
with open("cat_dog_classifier.pkl", "rb") as f:
    model = pickle.load(f)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def predict(image):
    img = load_img(image, target_size=(150, 150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    prediction = model.predict(img)
    return 'Dog' if prediction[0][0] > 0.5 else 'Cat'

st.title("Cat vs Dog Classifier")
st.write("Upload an image of a cat or a dog and the model will predict which one it is.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")
    label = predict(uploaded_file)
    st.write(f"The image is classified as: {label}")
