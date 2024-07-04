# HR Analytics: Job Change of Data Scientists

Dataset: https://www.kaggle.com/arashnic/hr-analytics-job-change-of-data-scientists

## Dependencies
- Python 3.x
- Required Libraries: Pandas, NumPy, Scikit-learn, LightGBM, Missingno, Matplotlib, Seaborn

## Usage
- Clone the Repository: git clone https://github.com/sujikathir/HR-Analytics.git
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

**Step 1: Data Splitting**

The training dataset is divided into two parts for validation:
- **train**: The main training set.
- **test_X**: A subset of the training set (first 1000 rows) used for validation, with the target variable dropped.

**Step 2: Merging Datasets**

Merge the `train` and `test_X` datasets with the `test` set that already exists.

**Step 3: Feature Transformation**

Perform feature transformation to prepare the data for modeling.

**Step 4: Handling Missing Values**

Use KNN imputer to fill missing values in the dataset.

**Step 5: Model Selection and Training**

Use the Light GBM algorithm for prediction due to its efficiency and speed, especially with large datasets.

**Step 6: Model Evaluation**

Evaluate the model using log-loss and roc-auc score to prevent overfitting and ensure correct classification.

**Step 7: Prediction**

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

- **Categorical columns**: 10 columns

- **Numerical columns**: 2 columns

- **Categorical but Cardinal columns**: 2 columns

- **Numerical but Categorical columns**: 2 columns

### Summary of Categorical Columns

The `cat_summary` function provides a summary of each categorical column, including the count and ratio of each category. For example:

- **Gender**:
  
  ![](https://github.com/sujikathir/HR-Analytics/blob/main/source/1.%20female%20vs%20male.png)
  
  - Male: 68.97%
  - Female: 6.46%
  - Other: 1.01%

This indicates that the majority of the observations are male.

- **Relevant Experience**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/2.%20has%20relevant%20exp.png)

  - Has relevant experience: 71.95%
  - No relevant experience: 28.05%

This shows that most individuals in the dataset have relevant experience.

- **Enrolled University**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/3.%20enrolled%20university.png)

  - No enrollment: 72.04%
  - Full-time course: 19.69%
  - Part-time course: 6.30%

This suggests that the majority of individuals are not enrolled in any university.

- **Target ratio for 0 vs 1**

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/9.%20target%20.png)

   - Target 0.0: 75.00%
   - Target 1.0: 25.00%
     
### Target Summary with Categorical Columns

The `target_summary_with_cat` function shows the relationship between the target variable and each categorical column. It provides the mean and count of the target variable for each category. For example:

- **Gender**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/1.1.%20female%20vs%20male.png)

  - Female: Target Mean = 0.261, Count = 1,171
  - Male: Target Mean = 0.228, Count = 12,532
  - Other: Target Mean = 0.258, Count = 178

This indicates that the average target value is slightly higher for females compared to males and others.

- **Relevant Experience**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/2.2.%20has%20relevant%20exp.png)

  - Has relevant experience: Target Mean = 0.215, Count = 13,072
  - No relevant experience: Target Mean = 0.341, Count = 5,086

This shows that individuals with no relevant experience have a higher average target value.

- **Enrolled University**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/3.3.%20enrolled%20university.png)

  - Full-time course: Target Mean = 0.383, Count = 3,566
  - Part-time course: Target Mean = 0.251, Count = 1,146
  - No enrollment: Target Mean = 0.212, Count = 13,087

This suggests that individuals enrolled in full-time courses have a higher average target value.

- **Education Level**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/4.4.%20graduate.png)

  - Graduate: Target Mean = 0.281, Count = 10,990
  - Masters: Target Mean = 0.214, Count = 4,146
  - High School: Target Mean = 0.198, Count = 1,913
  - Phd: Target Mean = 0.145, Count = 387
  - Primary School: Target Mean = 0.131, Count = 290

This indicates that individuals with a higher education level (Graduate) have a higher average target value, while those with a lower education level (Primary School) have a lower average target value.

- **Major Discipline**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/5.5.%20STEM.png)

  - STEM: Target Mean = 0.262, Count = 13,738
  - Humanities: Target Mean = 0.216, Count = 629
  - Other: Target Mean = 0.273, Count = 362
  - Business Degree: Target Mean = 0.266, Count = 312
  - Arts: Target Mean = 0.209, Count = 244
  - No Major: Target Mean = 0.256, Count = 211

This shows that individuals with a STEM major have a slightly higher average target value compared to those with other majors.

- **Company Size**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/6.6.%20Company%20size.png)

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

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/7.7.%20company%20type.png)

  - Pvt Ltd: Target Mean = 0.181, Count = 9,300
  - Funded Startup: Target Mean = 0.139, Count = 953
  - Public Sector: Target Mean = 0.223, Count = 902
  - Early Stage Startup: Target Mean = 0.242, Count = 571
  - NGO: Target Mean = 0.188, Count = 484
  - Other: Target Mean = 0.237, Count = 118

This shows that individuals working in early-stage startups have a higher average target value compared to those in other types of companies.

- **Last New Job**:

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/8.8.%20last%20new%20job.png)

  - 1 year: Target Mean = 0.266, Count = 7,620
  - 2 years: Target Mean = 0.240, Count = 2,751
  - 3 years: Target Mean = 0.224, Count = 973
  - 4 years: Target Mean = 0.223, Count = 972
  - Greater than 4 years: Target Mean = 0.182, Count = 3,118
  - Never: Target Mean = 0.306, Count = 2,318

This indicates that individuals who have never changed their job have a higher average target value compared to those who have changed jobs more recently.

The analysis helps in understanding the distribution of categorical features and their relationship with the target variable, which is crucial for handling imbalanced datasets and making informed decisions in the data preprocessing and modeling phases.

## Target Summary with Numerical Columns

This function calculates the average value of specified numerical columns grouped by the target variable.

### city_development_index

The `city_development_index` column shows the average value for each target class:

- **Target 0.0**: 0.853093
- **Target 1.0**: 0.756189

This indicates that the average `city_development_index` is higher for observations with target value 0.0 compared to those with target value 1.0.

### training_hours

The `training_hours` column shows the average value for each target class:

- **Target 0.0**: 66.205349
- **Target 1.0**: 63.076094

This suggests that the average `training_hours` are slightly higher for observations with target value 0.0 compared to those with target value 1.0.

## Missing Data Visualization

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/missing%20values.png)

This section outlines the extent of missing data within the dataset, highlighting variables with varying degrees of completeness.

### Variables with the Highest Proportion of Missing Data

- **company_type**: 31.82% missing

- **company_size**: 30.82% missing

- **gender**: 23.56% missing

### Variables with Moderate Levels of Missing Data

- **major_discipline**: 14.68% missing

- **target**: 14.70% missing

### Variables with Low Levels of Missing Data

- **last_new_job**: 2.18% missing

- **education_level**: 2.41% missing

- **enrolled_university**: 1.96% missing

- **experience**: 0.33% missing

### Variables with No Missing Data

- **enrollee_id**

- **city**

- **city_development_index**

- **relevent_experience**

- **training_hours**

This missing data pattern suggests that information related to companies (type and size) and personal attributes (gender, major discipline) are the most challenging to collect completely. The substantial amount of missing data in these fields could impact analyses and may require careful handling in any modeling or decision-making processes based on this dataset.

Understanding the distribution and extent of missing data is crucial for:

- **Data Cleaning**: Determining the appropriate strategies for imputation or removal.

- **Modeling**: Ensuring that models are robust to missing data or appropriately handle it.

- **Decision Making**: Recognizing the potential biases introduced by missing data and adjusting analyses accordingly.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/heatmap.png)

The heatmap shows the correlations between different categorical variables in the dataset. 

**Key observations:**

**Strong correlation:** The strongest correlation (0.8) is between company_size and company_type. This suggests that certain company types are strongly associated with specific company sizes.

**Moderate correlations:**

- Major_discipline and education_level (0.4): This indicates some relationship between a person's education level and their field of study.

- Major_discipline and experience (0.4): Suggests a link between a person's field of study and their work experience.

- Company_size and experience (0.3): Implies that work experience may be related to the size of the company a person works for.


**Weak correlations:** Most other variable pairs show weak correlations (0.1 or 0.2), indicating limited relationships between these factors.

- Gender correlations: Gender shows weak correlations (0.1) with most other variables, suggesting gender may not be strongly associated with other factors in this dataset.

- Last_new_job: This variable shows consistently weak correlations (0.1-0.2) with all other variables, indicating that the timing of a person's last job change might not be strongly related to other factors.

**Symmetry:** The heatmap is symmetrical, as correlation matrices typically are.

**No negative correlations:** All visible correlations are positive (blue shades), with no negative correlations (which would be shown in red) apparent.

This heatmap provides insights into the relationships between various categorical variables in the dataset, which could be useful for understanding patterns in employment, education, and company characteristics.

## Handling Multiclass Categorical Columns

To handle multiclass categorical columns, label encoding is applied before filling the missing values.

### Identifying Multiclass Categorical Columns

The code identifies multiclass categorical columns, which are columns with more than two unique values and a data type of object (categorical).

### Applying Label Encoding

Label encoding is applied to all identified multiclass categorical columns to convert categorical values into numerical values before performing any imputation.

## KNN Imputation and Model Evaluation

To find the optimal number of neighbors for KNN imputation, a function (`knn_imputer_test`) is used. This function tests different `n_neighbors` values and identifies the one that provides the best ROC AUC score using the LightGBM (LGBM) classifier.

### Function to Test KNN Imputer

The `knn_imputer_test` function performs the following steps:

1. **Data Preparation**: Creates a copy of the dataset and identifies columns with missing values.

2. **KNN Imputation**: Applies KNN imputation with different `n_neighbors` values to fill the missing values.

3. **Model Training and Evaluation**: Splits the imputed dataset into training and testing sets, upscales the data using SMOTE, trains a LightGBM model, and evaluates it using the ROC AUC score.

4. **Identifying the Best `n_neighbors`**: Tracks and returns the `n_neighbors` value that provides the highest ROC AUC score.

### Running the KNN Imputer Test

The function is executed to determine the best `n_neighbors` value for KNN imputation, aiming to maximize the ROC AUC score.

### Results of KNN Imputer Test

The function outputs the ROC AUC scores for different `n_neighbors` values and identifies the optimal `n_neighbors` value that achieves the highest ROC AUC score.

### Columns to be Filled

The columns identified for imputation are those with missing values excluding the target variable and the `enrollee_id`.

## Model Training and Evaluation After KNN Imputation

After completing the KNN imputation to fill in the missing values, the dataset is split into training and testing sets. The LGBM (LightGBM) model is then trained on the training set, and its performance is evaluated on both the training and testing sets.

### Training the Model

The LGBM model is fitted on the training set, allowing it to learn the patterns and relationships within the data. This involves:

- **Model Fitting**: The LGBM model is trained on the training data using the identified features and target variable.

- **Prediction**: The model makes predictions on both the training and testing sets.

- **Evaluation**: The performance of the model is evaluated using metrics such as precision, recall, F1-score, and ROC AUC score. This helps in understanding how well the model generalizes to unseen data.

## Plotting the Learning Curve

### Function to Plot Learning Curve

A function `plot_learning_curve` is defined to plot the logistic loss values for training and testing sets over different sizes of training data.

- **Iterative Training**: The model is trained on increasing sizes of the training data, and the logistic loss is calculated for both the training and testing sets.

- **Plotting**: The training and testing loss values are plotted to visualize the learning curve.

### Initial Learning Curve

The function is used to plot the learning curve for the initial dataset before any data upscaling is applied.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/training%20learning%20curve.png)

## Data Upscaling

To handle the class imbalance in the dataset, Synthetic Minority Over-sampling Technique (SMOTE) is applied multiple times to create a balanced dataset.

- **SMOTE Application**: SMOTE is applied three times, generating synthetic samples for the minority class.

- **Concatenation**: The resulting datasets from each SMOTE application are concatenated to form a final balanced dataset.

- **Class Distribution**: The final dataset has an equal number of samples for each class, ensuring balanced class distribution.

## Plotting the Learning Curve After Upscaling

The learning curve is plotted again using the upscaled dataset to observe any improvements in the model's performance.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/upscaled%20-%20training%20learning%20curve.png)

## Model Training and Evaluation

### Splitting the Data

The upscaled dataset is split into training and validation sets.

### Training the Model

The LGBM model is trained on the training set.

### Training Set Evaluation

- **Predictions and Probabilities**: The model's predictions and predicted probabilities are obtained for the training set.

- **Classification Report**: The classification report is generated, showing precision, recall, and F1-score for each class.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/training%20.png)

- **ROC AUC Score**: The ROC AUC score is calculated for the training set, indicating the model's performance.

### Validation Set Evaluation

- **Predictions and Probabilities**: The model's predictions and predicted probabilities are obtained for the validation set.

- **Classification Report**: The classification report is generated, showing precision, recall, and F1-score for each class.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/validation.png)

- **ROC AUC Score**: The ROC AUC score is calculated for the validation set, indicating the model's performance.

## Model Tuning

### Hyperparameter Tuning

A grid search is performed to tune the hyperparameters of the LGBM model using cross-validation.

- **Hyperparameter Grid**: A range of values for `num_leaves`, `max_depth`, `learning_rate`, and `n_estimators` is defined.

- **Grid Search**: The grid search is performed using 5-fold cross-validation to identify the best combination of hyperparameters.

- **Best Parameters**: The best hyperparameters are identified from the grid search results.

### Results of Hyperparameter Tuning

The best hyperparameters for the LGBM model are:
- `num_leaves`: 50
- `max_depth`: 8
- `learning_rate`: 0.02
- `n_estimators`: 1000

These hyperparameters are expected to optimize the model's performance.

## Final Model Training and Evaluation

After tuning the model, the final LGBM (LightGBM) model is trained with the best hyperparameters obtained from the GridSearchCV process. This final model is then evaluated on the validation set and the test set to ensure its robustness and accuracy.

### Training the Final Model

The final LGBM model is trained using the best hyperparameters identified during the model tuning process:

- **Model Fitting**: The LGBM model is trained on the training data with the optimal hyperparameters.

- **Prediction**: Predictions are made on the validation set to evaluate the model's performance.

### Validation Set Evaluation

The model's performance on the validation set is assessed using the following metrics:

- **Classification Report**: Precision, recall, F1-score, and support are computed to provide a detailed evaluation of the model's performance.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/final%20model%20validation.png)

- **ROC AUC Score**: The ROC AUC score is calculated to measure the model's ability to distinguish between the classes.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/ROC%20Curve%20-%20validation%20data.png)

- **ROC Curve Plot**: The ROC curve is plotted to visualize the trade-off between the true positive rate and the false positive rate.

### Test Set Evaluation

The final LGBM model is also evaluated on the test set to ensure its generalizability:

- **Prediction**: Predictions are made on the test data.

- **Classification Report**: The performance metrics are calculated for the test set.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/test.png)

- **ROC AUC Score**: The ROC AUC score is computed for the test set.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/ROC%20Curve%20-%20test%20data.png)

- **ROC Curve Plot**: The ROC curve for the test set is plotted.

- **Confusion Matrix**: The confusion matrix is plotted as a heatmap to visualize the model's performance in terms of actual vs. predicted labels.

![](https://github.com/sujikathir/HR-Analytics/blob/main/source/Confusion%20matrix.png)

## Conclusion

### Training and Validation Performance

The final LGBM model demonstrates strong performance on both the training and validation sets:

- **Training Set Results**:
  - Precision, recall, and F1-score for both classes are high, indicating a balanced performance in predicting both positive and negative classes.
  - The overall accuracy is 87%, and the ROC AUC score is 0.9438, showing the model's excellent ability to distinguish between classes on the training data.

- **Validation Set Results**:
  - The validation set results are slightly better than the training set, with an accuracy of 89% and a ROC AUC score of 0.9615.
  - This indicates that the model generalizes well to unseen data, maintaining a high level of performance.

### Test Set Performance

When applied to the test set, the model's performance is slightly lower:

- **Test Set Results**:
  - The overall accuracy is 80%, with a ROC AUC score of 0.8013.
  - The precision and recall for the positive class are lower compared to the training and validation sets, suggesting some difficulty in accurately predicting positive cases in the test data.

### Insights and Recommendations

1. **Model Robustness**: The model shows robust performance on the training and validation sets, indicating that the chosen features and hyperparameters are effective.

2. **Generalization**: The slightly lower performance on the test set suggests that while the model generalizes well, there may be some variability in new, unseen data. This could be due to differences in the distribution of the test data or the presence of more challenging cases.

3. **Imbalanced Data**: The precision and recall for the positive class on the test set indicate that further handling of class imbalance might be beneficial. Techniques such as more extensive use of SMOTE, ensemble methods, or cost-sensitive learning could be explored.

4. **Model Fine-Tuning**: Additional fine-tuning and testing with different hyperparameters, feature selection methods, or even different models could help improve the performance on the test set.

Overall, the final LGBM model provides a strong foundation for predicting the target variable, with excellent performance on the training and validation sets and good performance on the test set. Continuous monitoring and fine-tuning will help maintain and potentially improve its predictive accuracy in a production environment.



