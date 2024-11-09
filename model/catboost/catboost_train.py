import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
import os

# Load the data
data = pd.read_csv('./data/raw/train.csv')

# Preprocessing
# Drop columns that won’t contribute to model prediction
data = data.drop(['id', 'Name'], axis=1)

# Fill missing values for numerical features (if any)
data['Academic Pressure'].fillna(data['Academic Pressure'].median(), inplace=True)
data['Work Pressure'].fillna(data['Work Pressure'].median(), inplace=True)
data['CGPA'].fillna(data['CGPA'].median(), inplace=True)
data['Study Satisfaction'].fillna(data['Study Satisfaction'].median(), inplace=True)
data['Job Satisfaction'].fillna(data['Job Satisfaction'].median(), inplace=True)

# Handle missing values in categorical features by replacing NaN with a placeholder (e.g., 'Unknown')
categorical_features = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                        'Dietary Habits', 'Degree', 'Sleep Duration', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Replace NaN values in categorical columns with 'Unknown'
for col in categorical_features:
    data[col].fillna('Unknown', inplace=True)

# Encode binary categorical columns (e.g., 'Yes'/'No') to integers or strings if needed
data['Have you ever had suicidal thoughts ?'] = data['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0, 'Unknown': -1}).astype(float)
data['Family History of Mental Illness'] = data['Family History of Mental Illness'].map({'Yes': 1, 'No': 0, 'Unknown': -1}).astype(float)

# Ensure all categorical columns are strings
for col in categorical_features:
    data[col] = data[col].astype(str)

# Define target and features
X = data.drop('Depression', axis=1)
y = data['Depression'].astype(int)  # Ensure the target is integer-encoded

# Train/test split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CatBoost Pool (to handle categorical features)
train_pool = Pool(X_train, y_train, cat_features=categorical_features)
valid_pool = Pool(X_valid, y_valid, cat_features=categorical_features)

# Initialize and train CatBoost model
model = CatBoostClassifier(
    iterations=500,  # Can be increased for a better model if runtime allows
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    verbose=50
)

# Train the model
model.fit(train_pool, eval_set=valid_pool, use_best_model=True)

# Evaluate the model
y_pred = model.predict(X_valid)
y_proba = model.predict_proba(X_valid)[:, 1]
print(f"Accuracy: {accuracy_score(y_valid, y_pred)}")
print(f"F1 Score: {f1_score(y_valid, y_pred)}")
print(f"ROC AUC Score: {roc_auc_score(y_valid, y_proba)}")

# Save the model and related information to './model/catboost/catboost_info'
save_path = './model/catboost/catboost_info'
os.makedirs(os.path.dirname(save_path), exist_ok=True)
model.save_model(save_path)
print(f"Best CatBoost model and information saved to {save_path}")
