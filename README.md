# Exploring-Mental-Health-Data
This repo is for participating in [this](https://www.kaggle.com/competitions/playground-series-s4e11) Kaggle competition.

*"**Your Goal**: Use data from a mental health survey to explore factors that may cause individuals to experience depression."*

### Evaluation
Submissions are evaluated using <ins>Accuracy Score</ins>.

For each id row in the test set, you must predict the target Depression. The file should contain a header and have the following format:
``` xml
id,Depression
140700,0
140701,0
140702,1
etc.
```

## Setup
This repo uses a Conda environment configured in `environment.yml`. Here are the steps to set these up properly from this repos home folder:
1. Create an new Conda environment `conda env create -f environment.yml`
2. Activate the environment `conda activate Exploring-Mental-Health-Data`

If changes are made to `environment.yml` then update by running `conda env update --file environment.yml --prune`

## Commiting
Please refer to the Conventional Commits specification located [here](https://www.conventionalcommits.org/en/v1.0.0/) for structing your commit messages.

## File Manifest
All models are genetated in the `./model/<model-type>` folders. The goal is to try to solve this with many different strategies. Models can predict on the data by running the `./model/model_predict.py` script (after changing the path to the model pickle file). Predictions are automatically formatted for Kaggle and stored in `./submission`.

## TODO 
1. Data Exploration:
    - Perform and EDA on the given dataset
2. Data Pre-Processing & Feature Engineering:
    - Handle missing values via impution or dropping
    - Address class imbalanced with techniques like SMOTE, undersampling, or weighting
    - Create new features if useful
    - Perform encoding on categorical features
    - Optionally bin numerical data
3. Model Development:
    - Build decision tree model in XGBoost to leverage GPU support
    - Build a CatBoost model
    - Build Logistic regression model
    - Build Support Vector Machine model
    - Build Neural Net Model
