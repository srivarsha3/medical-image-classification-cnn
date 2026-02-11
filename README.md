# Medical Image Classification using CNN

## Project Overview
This project demonstrates how a Convolutional Neural Network (CNN) can be used to classify
chest X-ray images into **Normal** and **Pneumonia** categories.  
The model helps in understanding how deep learning can assist medical diagnosis.

---

##  Objective
To build a deep learning model that automatically classifies medical images
and supports faster and accurate disease detection.

---

##  Problem Statement
Manual analysis of medical images is time-consuming and prone to human error.
This project aims to automate the classification process using machine learning.

---

##  Dataset
- Chest X-ray images (Normal and Pneumonia)
- Publicly available medical dataset (e.g., Kaggle)
- Dataset is organized into training and testing folders

📌 *Note: Dataset is not uploaded due to size limitations.*

---

##  Preprocessing Steps
- Image resizing
- Grayscale conversion
- Normalization of pixel values
- Train-test split

---

##  Model Used
- Convolutional Neural Network (CNN)
- Automatically extracts features from images
- Suitable for medical image classification

---

## Evaluation Metrics
- Accuracy
- Precision
- Recall

**Achieved Accuracy:** ~92%

---

##  Tools & Technologies
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

---

## How to Run the Project
1. Clone the repository  
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Web Application Output

The trained CNN model is deployed using a Flask web application. 
The application allows users to upload chest X-ray images and predicts whether the image indicates Pneumonia or Normal condition.


<img width="1439" height="793" alt="1" src="https://github.com/user-attachments/assets/2732a152-5753-4930-bd5c-de67e4860ff2" />


### Output Screenshots

- Home page with image upload option
- Uploaded chest X-ray image
- Prediction result with classification label and confidence score

