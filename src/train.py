
# train.py
"""
This file trains and evaluates the CNN model.
"""

from sklearn.model_selection import train_test_split
from preprocessing import load_images
from model import build_model

DATASET_DIR = "dataset/train"
CATEGORIES = ["NORMAL", "PNEUMONIA"]
IMG_SIZE = 128

X, y = load_images(DATASET_DIR, CATEGORIES, IMG_SIZE)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = build_model(IMG_SIZE)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)
