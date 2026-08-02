# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset-Completion-requirements

## AIM
Build a binary classification model using a neural network to predict whether an individual earns more than $50K annually based on demographic and work-related features.

## Input Features
#### Continuous Features (cont_cols)
age
hours-per-week

#### Categorical Features (cat_cols)
Typical Census Income dataset categorical variables include:
workclass
education
marital-status
occupation
relationship
race
sex
native-country

#### Target Column (label_col)
label → Binary class:
<=50K → class 0
>50K → class 1

## Model Steps:
#### 1. Data Loading
Load the income.csv dataset (30,000 entries) using Pandas.
Inspect the dataset to understand the features and target variable.

#### 2. Data Preprocessing
Separate columns into:

cat_cols: Categorical features (sex, education, etc.)
cont_cols: Continuous features (age, hours-per-week)
y_col: Label column (label)
Convert all categorical columns to category dtype for encoding.
Embedding Setup
Determine the number of unique categories in each categorical column.
Define embedding sizes using the rule: (category_size, min(50, (category_size + 1)//2))
Data Conversion
Convert categorical columns to category codes and stack into a NumPy array.

Convert continuous columns to a NumPy array.

#### 3. Convert both arrays into PyTorch tensors:

cats: Categorical tensor (int64)
conts: Continuous tensor (float32)
y: Label tensor (flattened)
Train/Test Split

#### 4. Split the data into:

Training set (25,000 records)
Testing set (5,000 records)
Model Architecture

#### 5. Create a custom PyTorch model class TabularModel that:

Handles embedding layers for categorical data.
Applies batch normalization and dropout.
Uses one or more fully connected layers for prediction.
Final output layer has 2 units (binary classification).

Training Setup

#### 6. Define:

criterion: CrossEntropyLoss
optimizer: Adam optimizer with lr=0.001
Set random seed for reproducibility.

#### 7. Model Training
Train for 300 epochs.
Track and store loss values per epoch.
Loss Visualization
Plot CrossEntropy Loss vs Epochs to observe convergence.
Model Evaluation

#### 8. Evaluate on the test set:

Calculate Cross Entropy Loss
Compute accuracy of predictions

## Result:
Thus the program was successfully executed
