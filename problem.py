import os
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

# Problem definition
problem_title = 'Prediction of next-day rain in Australia'
_target_column_name = 'RainTomorrow'
_prediction_label_names = [0, 1]  # 0: No rain, 1: Rain

# Define the prediction type
# We're dealing with a binary classification task
Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names
)

# Define the workflow
workflow = rw.workflows.Estimator()

# Define score metrics
score_types = [
    rw.score_types.Accuracy(name='acc', precision=4),
    rw.score_types.ROCAUC(name='roc_auc', precision=4),
    rw.score_types.F1Above(name='f1', precision=4),
    rw.score_types.NegativeLogLikelihood(name='nll', precision=4),
]

def get_cv(X, y):
    """Return cross-validation splits for training."""
    cv = StratifiedShuffleSplit(n_splits=8, test_size=0.2, random_state=42)
    return cv.split(X, y)

def _read_data(path, df_filename):
    # Read the CSV file
    data_path = os.path.join(path, 'data', df_filename)
    df = pd.read_csv(data_path)
    
    # Process target column
    # Map 'Yes' to 1 and 'No' to 0
    target_mapping = {'No': 0, 'Yes': 1}
    
    if _target_column_name in df.columns:
        y_array = df[_target_column_name].map(target_mapping).values
        X_df = df.drop(_target_column_name, axis=1)
    else:
        y_array = None
        X_df = df
    
    return X_df, y_array

def get_train_data(path='.'):
    return _read_data(path, 'train.csv')

def get_test_data(path='.'):
    return _read_data(path, 'test.csv')