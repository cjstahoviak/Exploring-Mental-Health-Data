import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Load the data
raw_data = pd.read_csv('./data/raw/train.csv')

# Drop columns that won’t contribute to model prediction
raw_data = raw_data.drop(['id', 'Name'], axis=1)

# Create dataset with only complete rows
dropped_data = raw_data.dropna()

# Create dataset with missing values imputed using the mean
imputed_avg_data = raw_data.copy()  # Copy the raw data to retain original structure
imputed_avg_data.fillna(imputed_avg_data.mean(numeric_only=True), inplace=True)

# Create dataset with missing values imputed using the median
imputed_median_data = raw_data.copy()  # Copy the raw data again
imputed_median_data.fillna(imputed_median_data.median(numeric_only=True), inplace=True)

# Define target and features
categorical_features = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                        'Dietary Habits', 'Degree', 'Sleep Duration', 
                        'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

for col in categorical_features:
    # Use direct assignment to modify the DataFrame
    imputed_avg_data[col] = imputed_avg_data[col].fillna('Unknown')
    imputed_median_data[col] = imputed_median_data[col].fillna('Unknown')

    # Ensure all categorical columns are strings
    imputed_avg_data[col] = imputed_avg_data[col].astype(str)
    imputed_median_data[col] = imputed_median_data[col].astype(str)

# Save processed datasets into csv files
os.makedirs('./data/processed', exist_ok=True)  # Ensure the directory exists
dropped_data.to_csv('./data/processed/train_dropped.csv', index=False)
imputed_avg_data.to_csv('./data/processed/train_imputed_average.csv', index=False)
imputed_median_data.to_csv('./data/processed/train_imputed_median.csv', index=False)

print("Processed datasets saved to './data/processed/'")