# Title
Chicken Swarm Optimization with Hybrid Network-Based Oral Cancer Classification in a Distributed Cloud Environment

# Description
This repository contains the implementation of the proposed CSOHN-OCC model for oral cancer classification. The model combines Gaussian filtering, HybridNet-based feature extraction, Chicken Swarm Optimization-based hyperparameter tuning, and Deep Stacked Sparse Autoencoder-based classification. The framework is designed for oral cancer image classification in a distributed cloud environment to support remote diagnostic applications.


# Dataset Information
The study uses oral cancer histopathological image data consisting of two classes:
Class	Number of Images
Non-Cancer	439
Cancer	89
Total	528
Dataset link:
https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images



# Code Information
The code includes the following modules:
preprocessing.py – Image resizing, normalization, and Gaussian filtering.
feature_extraction.py – HybridNet-based deep feature extraction.
cso_optimization.py – Chicken Swarm Optimization for hyperparameter tuning.
dssae_classifier.py – Deep Stacked Sparse Autoencoder classifier.
train.py – Model training and validation.
evaluate.py – Performance evaluation using accuracy, precision, recall, F1-score, and MCC.
utils.py – Utility functions for dataset loading, plotting, and metric calculation.


# Usage Instructions
Clone or download the repository.
Download the oral cancer dataset from the provided Kaggle link.
Place the dataset inside the dataset/ folder using the following structure:
dataset/
 ├── cancer/
 └── non_cancer/
Install the required Python libraries.
Run preprocessing and training:
python train.py
Evaluate the trained model:
python evaluate.py
The output will include classification accuracy, precision, recall, F1-score, MCC, confusion matrix, training/validation accuracy curve, and training/validation loss curve.

# Requirements
The implementation requires the following software and libraries:
Python 3.8 or above
NumPy
Pandas
OpenCV
Matplotlib
Scikit-learn
TensorFlow / Keras
SciPy
Pillow
Install dependencies using:
pip install numpy pandas opencv-python matplotlib scikit-learn tensorflow scipy pillow



# Methodology
The methodology follows these steps:
Dataset Loading
Oral cancer images are loaded from cancer and non-cancer folders.
Image Preprocessing
Images are resized, normalized, and enhanced using Gaussian filtering to remove noise.
Feature Extraction
HybridNet is used to extract discriminative deep image features from oral cancer images.
Hyperparameter Optimization
Chicken Swarm Optimization is applied to tune important model parameters and improve classification performance.
Classification
The optimized features are classified using a Deep Stacked Sparse Autoencoder.
Performance Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, Matthews Correlation Coefficient, confusion matrix, and precision-recall analysis.


# Citations

This dataset was used for oral cancer classification experiments:

https://www.kaggle.com/datasets/shivam17299/oral-cancer-lips-and-tongue-images
