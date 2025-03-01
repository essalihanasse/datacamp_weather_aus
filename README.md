# Australian Rain Prediction - Code Documentation

This document provides detailed documentation for the codebase of the Australian Rain Prediction project. Each file's purpose and functionality is explained to help you understand and extend the project.

## Project Structure

```
.
├── data/
│   ├── weatherAUS.csv          
│   ├── train.csv               
│   └── test.csv               
├──submissions/
|   └── strarting_kit/
|       └──estimator.py
├── problem.py                  
├── requirements.txt            
└── README.md                   
```

## Files Explained

### `problem.py`

Defines the RAMP workflow structure and interfaces:

- `problem_title`: Descriptive title of the prediction task
- `_target_column_name`: Identifies 'RainTomorrow' as the prediction target
- `_prediction_label_names`: Binary classification with [0, 1] labels
- `Predictions`: Custom class for prediction objects
- `workflow`: Uses the standard RAMP Estimator workflow
- `score_types`: Defines evaluation metrics (Accuracy, ROC AUC, F1, NLL)
- `get_cv()`: Creates cross-validation splits
- `_read_data()`: Reads and processes CSV files
- `get_train_data()`, `get_test_data()`: Public data access methods


### `submissions.py`

Standard submission implementation:

- `FeatureExtractor`: Class that transforms raw features:
  - Handles numerical and categorical features
  - Performs date parsing and feature extraction
  - Engineers weather differentials (temperature, pressure, etc.)
  - Creates the preprocessing pipeline with scikit-learn

- `get_estimator()`: Returns the complete pipeline with:
  - Feature extraction step
  - Gradient Boosting classifier with optimized parameters

### `evaluate_models.py`

Comprehensive evaluation framework:

- `evaluate_submission()`: Evaluates a single model:
  - Loads the model and data
  - Times training and prediction
  - Calculates performance metrics
  - Generates visualizations (confusion matrix, ROC curve, PR curve)

- `compare_submissions()`: Compares multiple models:
  - Evaluates each model
  - Creates comparison tables
  - Generates comparative visualizations

### `requirements.txt`

Lists all required Python packages with version constraints for reproducibility.
