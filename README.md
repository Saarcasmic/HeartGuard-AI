# HeartGuard AI  
### Heart Disease Prediction using Machine Learning

---

Heart disease prevention has become essential for public health. Data-driven systems for predicting heart disease can greatly enhance research and preventive care, helping more people live healthier lives. This is where **Machine Learning** and **Artificial Intelligence** come into play, making accurate predictions for heart disease risk possible.

## Project Overview

**HeartGuard AI** analyzes patient data to predict heart disease risk. The project includes:
- **Data Processing**: Comprehensive data cleaning and feature engineering.
- **Model Training**: Training various machine learning models.
- **Prediction Accuracy**: Accurate predictions using advanced algorithms.

The project was developed in a Jupyter Notebook, with the dataset sourced from Kaggle’s [UCI Heart Disease dataset](https://www.kaggle.com/ronitf/heart-disease-uci).

## Objective

The goal is to predict the presence of heart disease in patients based on several health parameters. This is a **binary classification problem**:
- **Input Features**: Various health metrics and lifestyle parameters.
- **Target Variable**: Binary output indicating the presence or absence of heart disease.

## Machine Learning Algorithms Used

We implemented a range of machine learning algorithms in **Python** to build the prediction model:

- **Logistic Regression** (Scikit-learn)
- **Naive Bayes** (Scikit-learn)
- **Support Vector Machine (SVM)** (Scikit-learn)
- **K-Nearest Neighbors (KNN)** (Scikit-learn)
- **Decision Tree** (Scikit-learn)
- **Random Forest** (Scikit-learn)
- **XGBoost** (Scikit-learn)
- **Artificial Neural Network** (Keras) - Single hidden layer

**Best Accuracy Achieved**: 95% (using Random Forest)

## Project Workflow

1. **Data Preprocessing**: Cleaned and prepared data for model training.
2. **Feature Engineering**: Extracted meaningful features using TF-IDF and other techniques.
3. **Model Training**: Tested multiple algorithms to optimize predictive accuracy.
4. **Evaluation**: Evaluated models to identify the best-performing one (Random Forest).

## Dataset Information

- **Dataset Source**: Kaggle [UCI Heart Disease dataset](https://www.kaggle.com/ronitf/heart-disease-uci)
- **Data Characteristics**: Includes multiple attributes related to patient health, such as age, cholesterol levels, blood pressure, etc.

## Results

**HeartGuard AI** achieved a maximum prediction accuracy of **95%** with the **Random Forest** algorithm, demonstrating strong potential for real-world application in predictive healthcare systems.

---
