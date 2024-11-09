import pandas as pd
from catboost import CatBoostClassifier, Pool
import os

# Define the model path and submission path
model_path = './saved_model/catboost_best'
submission_dir = './submission'

# Load the model
model = CatBoostClassifier()
model.load_model(model_path)

# Load the test data
test_data = pd.read_csv('./data/raw/test.csv')

# Preprocessing the test data
# Drop columns that won't contribute to the prediction
test_data_cleaned = test_data.drop(['id', 'Name'], axis=1)

# Handle missing values for numerical features
test_data_cleaned['Academic Pressure'].fillna(test_data_cleaned['Academic Pressure'].median(), inplace=True)
test_data_cleaned['Work Pressure'].fillna(test_data_cleaned['Work Pressure'].median(), inplace=True)
test_data_cleaned['CGPA'].fillna(test_data_cleaned['CGPA'].median(), inplace=True)
test_data_cleaned['Study Satisfaction'].fillna(test_data_cleaned['Study Satisfaction'].median(), inplace=True)
test_data_cleaned['Job Satisfaction'].fillna(test_data_cleaned['Job Satisfaction'].median(), inplace=True)

# Handle missing values in categorical features by replacing NaN with a placeholder (e.g., 'Unknown')
categorical_features = ['Gender', 'City', 'Working Professional or Student', 'Profession',
                        'Dietary Habits', 'Degree', 'Sleep Duration', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']

# Replace NaN values in categorical columns with 'Unknown'
for col in categorical_features:
    test_data_cleaned[col].fillna('Unknown', inplace=True)

# Encode binary categorical columns (e.g., 'Yes'/'No') to integers or strings if needed
test_data_cleaned['Have you ever had suicidal thoughts ?'] = test_data_cleaned['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0, 'Unknown': -1}).astype(float)
test_data_cleaned['Family History of Mental Illness'] = test_data_cleaned['Family History of Mental Illness'].map({'Yes': 1, 'No': 0, 'Unknown': -1}).astype(float)

# Ensure all categorical columns are strings
for col in categorical_features:
    test_data_cleaned[col] = test_data_cleaned[col].astype(str)

# Prepare the feature set (X_test) for prediction
X_test = test_data_cleaned

# Create a Pool for prediction, specifying categorical features
test_pool = Pool(X_test, cat_features=categorical_features)

# Make predictions using the trained model
predictions = model.predict(test_pool)

# Prepare the final submission DataFrame
submission = pd.DataFrame({
    'id': test_data['id'],
    'Depression': predictions.astype(int)  # Ensure the predictions are integers (0 or 1)
})

# Define the output submission file name based on the model's name
submission_file = f"{os.path.basename(model_path)}_sub_1.csv"

# Save the submission file
submission_path = os.path.join(submission_dir, submission_file)
os.makedirs(submission_dir, exist_ok=True)
submission.to_csv(submission_path, index=False)

print(f"Submission file saved to {submission_path}")
