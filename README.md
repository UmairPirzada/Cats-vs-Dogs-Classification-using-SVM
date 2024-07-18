# Cats vs Dogs Classification using SVM
### visualize cat images

 ![image](https://github.com/user-attachments/assets/d0f68100-f16b-4ad1-983a-97c92f3a4e36)
 
### visualize dog images

 ![image](https://github.com/user-attachments/assets/b8bf562f-5a07-46e2-bb15-a4fd9b2e3a69)
 
### visualize both cat and dog images

![image](https://github.com/user-attachments/assets/053ea4d6-66cf-4985-a0bb-a615b571b0ab)

## Overview
This repository contains a Jupyter Notebook implementing a Support Vector Machine (SVM) classifier to distinguish between images of cats and dogs. The project leverages popular machine learning libraries and techniques to achieve accurate classification results.

## Features
- Data preprocessing and augmentation
- SVM model implementation
- Training and evaluation of the model
- Visualization of results

## Dataset
The dataset used for this project is the [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/competitions/dogs-vs-cats). It consists of 25,000 images of cats and dogs, split equally.

## Installation
To run the notebook, you need to have Python and Jupyter Notebook installed.

# Usage
### Clone this repository:
git clone https://github.com/UmairPirzada/PRODIGY_ML_03.git
# Open the Jupyter Notebook:
jupyter notebook Prodigy-ML-03.ipynb
Notebook Contents
The notebook includes the following sections:

### Importing Required Libraries:
Importing essential libraries like NumPy, pandas, scikit-learn, OpenCV, etc.
### Extracting the Datasets: 
Extracting the dataset from zip files.

### Loading and Preprocessing the Data:
Loading and normalizing images, converting them to NumPy arrays.

### Data Visualization:
Visualizing sample images from the dataset.

### Data Preparation:
Flattening images, standardizing features, and applying t-SNE for visualization.

### Model Training: 
Splitting the data into training and test sets, training the SVM model.

### Model Evaluation: 
Evaluating the model using accuracy, classification reports, and confusion matrices.

### Saving and Loading Models: 
Saving the trained SVM model and the scaler.

### Making Predictions:
Making predictions on the test set and visualizing some of the test results.

# Results
The final model's performance is evaluated using accuracy, precision, recall, and F1-score. Confusion matrices and classification reports are also provided for a comprehensive analysis of the model's performance.

# Contributing
Contributions are welcome! Please fork this repository and submit a pull request with your improvements.
