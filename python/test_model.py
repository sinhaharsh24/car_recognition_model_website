# === test_model.py ===
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# === Load model and class names ===
model = load_model('car_model.h5')

with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

IMAGE_SIZE = 128

# === Load and preprocess the test image ===
def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

# === Predict and display ===
def predict_image(image_path):
    img, img_array = load_image(image_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]

    print(f"Predicted Class Index: {predicted_class}")
    print(f"Predicted Label: {predicted_label}")

    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label}")
    plt.axis('off')
    plt.show()

# === Run prediction if image path is given ===
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <path_to_image>")
    else:
        predict_image(sys.argv[1])
