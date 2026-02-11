import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 128
model = load_model("models/pneumonia_model.h5")

def predict_pneumonia(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return f"PNEUMONIA DETECTED (Confidence: {prediction:.2f})"
    else:
        return f"NORMAL (Confidence: {1 - prediction:.2f})"

