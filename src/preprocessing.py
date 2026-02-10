# preprocessing.py
"""
This file contains functions for loading and preprocessing medical images.
"""

import cv2
import os
import numpy as np

def load_images(dataset_dir, categories, img_size):
    data = []
    labels = []

    for category in categories:
        path = os.path.join(dataset_dir, category)
        label = categories.index(category)

        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                img_array = cv2.resize(img_array, (img_size, img_size))
                data.append(img_array)
                labels.append(label)
            except:
                pass

    X = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
    y = np.array(labels)
    return X, y

