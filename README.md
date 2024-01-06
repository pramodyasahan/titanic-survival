# Titanic Survival Prediction Model

## Overview
This repository contains a machine learning project focused on predicting the survival of passengers on the Titanic. The project uses a Support Vector Regression (SVR) model from the sklearn library and involves data preprocessing and prediction.

## Data
The model uses datasets from the Titanic Kaggle competition:
- `train.csv`: Training data with passenger features and survival status.
- `test.csv`: Testing data with passenger features.

## Features
The datasets include passenger features like class, sex, and age.

## Preprocessing
Key preprocessing steps include:
- One-hot encoding of categorical variables (sex).
- Imputation of missing values in numerical features (age) using the mean strategy.
- The preprocessing is applied to both training and test data.

## Model Training
The SVR model with RBF kernel is trained using the preprocessed training data.

## Prediction
- The trained model predicts the survival status for the test dataset.
- Predictions are converted to binary format (0 for non-survival, 1 for survival) based on a threshold of 0.5.

## Output
- The predictions are saved into a CSV file named `pred.csv`.
- The CSV file includes PassengerId and the predicted survival status.

## Usage
To run the model:
1. Load the training and testing datasets.
2. Preprocess the data by encoding categorical features and handling missing values.
3. Train the SVR model using the training data.
4. Predict survival status for the test data.
5. Save the predictions in a CSV file.

## Dependencies
- pandas
- numpy
- matplotlib
- scikit-learn

## Note
This project is a demonstration of basic machine learning techniques applied to a historical dataset. The model and preprocessing steps can be further optimized and expanded for better performance or different datasets.
