# model.py
"""
This file defines the CNN model for medical image classification.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def build_model(img_size):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    return model

