# HR Analytics: Job Change of Data Scientists

Dataset: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

# HR Analytics: Job Change of Data Scientists

## Table of Contents
1. [Introduction](#introduction)
2. [Libraries and Their Uses](#libraries-and-their-uses)
3. [Business Problem](#business-problem)
4. [Project Goal](#project-goal)
5. [Dataset and Features](#dataset-and-features)
6. [Approach](#approach)
7. [Initial Data Exploration](#initial-data-exploration)

## Introduction
This project aims to predict whether candidates who have completed certain training courses are likely to work for the company that provided the training. This is a classic supervised machine learning problem where the target variable is binary (0 or 1).

## Libraries and Their Uses

### Data Manipulation and Analysis
- **numpy**: Used for numerical operations and handling arrays.

- **pandas**: Used for data manipulation and analysis, particularly for working with data frames (tables).

### Visualization
- **matplotlib**: A plotting library for creating static, interactive, and animated visualizations.

- **seaborn**: A statistical data visualization library based on matplotlib, used for making plots more attractive and informative.

- **missingno**: Used for visualizing missing values in a dataset.

### Machine Learning and Preprocessing
- **sklearn.preprocessing.LabelEncoder**: Used for encoding categorical features as numbers.

- **sklearn.impute.KNNImputer**: Used for imputing missing values using k-Nearest Neighbors.

- **sklearn.neighbors.LocalOutlierFactor**: Used for identifying outliers in the dataset.

- **sklearn.model_selection.train_test_split**: Used for splitting the dataset into training and testing sets.

- **sklearn.model_selection.GridSearchCV**: Used for hyperparameter tuning by performing a grid search.

- **sklearn.metrics**: Contains metrics for model evaluation, including roc_auc_score, roc_curve, log_loss, classification_report, and confusion_matrix.

- **lightgbm.LGBMClassifier**: A gradient boosting framework that uses tree-based learning algorithms.

- **imblearn.over_sampling.SMOTE**: Used for handling imbalanced datasets by oversampling the minority class.

## Business Problem
A company active in Big Data and Data Science wants to hire data scientists from people who have successfully passed their training courses. The goal is to identify which candidates are likely to work for the company after completing the training, as this helps in reducing costs and time, improving the quality of training, and better planning of courses and candidate categorization.

## Project Goal
The main objective of this project is to predict the probability that a candidate will work for the company after completing the training.

## Dataset and Features
The dataset contains the following features:

- **enrollee_id**: Unique ID for the candidate

- **city**: City code

- **city_development_index**: Development index of the city (scaled)

- **gender**: Gender of the candidate

- **relevant_experience**: Relevant experience of the candidate

- **enrolled_university**: Type of university course enrolled in, if any

- **education_level**: Education level of the candidate

- **major_discipline**: Major discipline of the candidate's education

- **experience**: Candidate's total experience in years

- **company_size**: Number of employees in the candidate's current company

- **company_type**: Type of current employer

- **last_new_job**: Years since the candidate's previous job

- **training_hours**: Training hours completed

- **target**: 0 if not looking for a job change, 1 if looking for a job change (only in the training set)

### Data Types
- `int64`: enrollee_id, training_hours
- `float64`: city_development_index
- `object`: city, gender, relevant_experience, enrolled_university, education_level, major_discipline, experience, company_size, company_type, last_new_job

## Approach

### Step 1: Data Splitting
The training dataset is divided into two parts for validation:
- **train**: The main training set.
- **test_X**: A subset of the training set (first 1000 rows) used for validation, with the target variable dropped.

### Step 2: Merging Datasets
Merge the `train` and `test_X` datasets with the `test` set that already exists.

### Step 3: Feature Transformation
Perform feature transformation to prepare the data for modeling.

### Step 4: Handling Missing Values
Use KNN imputer to fill missing values in the dataset.

### Step 5: Model Selection and Training
Use the Light GBM algorithm for prediction due to its efficiency and speed, especially with large datasets.

### Step 6: Model Evaluation
Evaluate the model using log-loss and roc-auc score to prevent overfitting and ensure correct classification.

### Step 7: Prediction
Predict the probability of a candidate working for the company on the test set.

## Initial Data Exploration
The initial exploration of the training and test datasets involves checking the shape, data types, and missing values, as well as summarizing the quantiles of numerical features.
