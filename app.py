
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model and class names
model = tf.keras.models.load_model("car_model")
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMAGE_SIZE = 128

def preprocess_image(img):
    img = img.convert("RGB")
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI

st.markdown("""
    <style>
    .main {
        background-color: #f7f7f7;
        padding: 20px;
    }
    h1 {
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)


st.title("🚗 Car Brand Classifier")
st.write("Upload an image of a car, and the model will predict its brand.")
st.write("only predict - Audi, Hyundai Creta, Mahindra Scorpio, Rolls Royce, Swift, Tata Safari, Toyota Innova")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Prediction: **{predicted_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence:.2f}%**")
