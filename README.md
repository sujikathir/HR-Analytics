# HR Analytics: Job Change of Data Scientists

Dataset: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

## Dependencies
- Python 3.x
- Required Libraries: Pandas, NumPy, Scikit-learn, LightGBM, Missingno, Matplotlib, Seaborn

## Usage
- Clone the Repository: git clone https://github.com/your_username/hr-analytics.git
- Install Dependencies: pip install -r requirements.txt
- Run Jupyter Notebooks: Navigate to notebooks and run jupyter notebook to execute the analysis and model building notebooks.


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

# Imbalanced Dataset Analysis and Summary

This project involves analyzing an imbalanced dataset, identifying categorical and numerical features, and summarizing the data using various functions. The goal is to understand the distribution of the target variable and its relationship with other features.

## Dataset Overview

The dataset has the following characteristics:
- **Observations**: 21,287
- **Variables**: 14
- **Imbalance in Target Variable**:
  - 0.0: 63.94%
  - 1.0: 21.36%

## Column Categories

Columns are categorized into categorical, numerical, categorical but cardinal, and numerical but categorical.

### Grabbing Column Names

The `grab_col_names` function identifies and categorizes columns based on their data types and the number of unique values.

- **Categorical Columns**: These columns contain categorical data, such as 'gender' and 'relevent_experience'.
- **Numerical Columns**: These columns contain numerical data, such as 'city_development_index' and 'training_hours'.
- **Categorical but Cardinal Columns**: These are categorical columns with a high number of unique values, such as 'city' and 'experience'.
- **Numerical but Categorical Columns**: These are numerical columns that have a low number of unique values, such as 'enrollee_id' and 'target'.

After running the function, the following column counts were identified:

- **Categorical columns**: 10 columns, e.g., 'gender', 'relevent_experience'
- **Numerical columns**: 2 columns, e.g., 'city_development_index', 'training_hours'
- **Categorical but Cardinal columns**: 2 columns, e.g., 'city', 'experience'
- **Numerical but Categorical columns**: 2 columns, e.g., 'enrollee_id', 'target'

### Summary of Categorical Columns

The `cat_summary` function provides a summary of each categorical column, including the count and ratio of each category. For example:

- **Gender**:
  - Male: 68.97%
  - Female: 6.46%
  - Other: 1.01%

This indicates that the majority of the observations are male.

- **Relevant Experience**:
  - Has relevant experience: 71.95%
  - No relevant experience: 28.05%

This shows that most individuals in the dataset have relevant experience.

- **Enrolled University**:
  - No enrollment: 72.04%
  - Full-time course: 19.69%
  - Part-time course: 6.30%

This suggests that the majority of individuals are not enrolled in any university.

### Target Summary with Categorical Columns

The `target_summary_with_cat` function shows the relationship between the target variable and each categorical column. It provides the mean and count of the target variable for each category. For example:

- **Gender**:
  - Female: Target Mean = 0.261, Count = 1,171
  - Male: Target Mean = 0.228, Count = 12,532
  - Other: Target Mean = 0.258, Count = 178

This indicates that the average target value is slightly higher for females compared to males and others.

- **Relevant Experience**:
  - Has relevant experience: Target Mean = 0.215, Count = 13,072
  - No relevant experience: Target Mean = 0.341, Count = 5,086

This shows that individuals with no relevant experience have a higher average target value.

- **Enrolled University**:
  - Full-time course: Target Mean = 0.383, Count = 3,566
  - Part-time course: Target Mean = 0.251, Count = 1,146
  - No enrollment: Target Mean = 0.212, Count = 13,087

This suggests that individuals enrolled in full-time courses have a higher average target value.

- **Education Level**:
  - Graduate: Target Mean = 0.281, Count = 10,990
  - Masters: Target Mean = 0.214, Count = 4,146
  - High School: Target Mean = 0.198, Count = 1,913
  - Phd: Target Mean = 0.145, Count = 387
  - Primary School: Target Mean = 0.131, Count = 290

This indicates that individuals with a higher education level (Graduate) have a higher average target value, while those with a lower education level (Primary School) have a lower average target value.

- **Major Discipline**:
  - STEM: Target Mean = 0.262, Count = 13,738
  - Humanities: Target Mean = 0.216, Count = 629
  - Other: Target Mean = 0.273, Count = 362
  - Business Degree: Target Mean = 0.266, Count = 312
  - Arts: Target Mean = 0.209, Count = 244
  - No Major: Target Mean = 0.256, Count = 211

This shows that individuals with a STEM major have a slightly higher average target value compared to those with other majors.

- **Company Size**:
  - 10/49: Target Mean = 0.236, Count = 1,389
  - 100-500: Target Mean = 0.160, Count = 2,420
  - 1000-4999: Target Mean = 0.153, Count = 1,266
  - 10000+: Target Mean = 0.195, Count = 1,900
  - 50-99: Target Mean = 0.175, Count = 2,919
  - 500-999: Target Mean = 0.172, Count = 837
  - <10: Target Mean = 0.172, Count = 1,254
  - 5000-9999: Target Mean = 0.183, Count = 530

This indicates that individuals working in smaller companies (10-49 employees) have a higher average target value compared to those in larger companies (100-500 employees).

- **Company Type**:
  - Pvt Ltd: Target Mean = 0.181, Count = 9,300
  - Funded Startup: Target Mean = 0.139, Count = 953
  - Public Sector: Target Mean = 0.223, Count = 902
  - Early Stage Startup: Target Mean = 0.242, Count = 571
  - NGO: Target Mean = 0.188, Count = 484
  - Other: Target Mean = 0.237, Count = 118

This shows that individuals working in early-stage startups have a higher average target value compared to those in other types of companies.

- **Last New Job**:
  - 1 year: Target Mean = 0.266, Count = 7,620
  - 2 years: Target Mean = 0.240, Count = 2,751
  - 3 years: Target Mean = 0.224, Count = 973
  - 4 years: Target Mean = 0.223, Count = 972
  - >4 years: Target Mean = 0.182, Count = 3,118
  - Never: Target Mean = 0.306, Count = 2,318

This indicates that individuals who have never changed their job have a higher average target value compared to those who have changed jobs more recently.

The analysis helps in understanding the distribution of categorical features and their relationship with the target variable, which is crucial for handling imbalanced datasets and making informed decisions in the data preprocessing and modeling phases.
