import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("========== 1. Loading and Exploring the Data ==========")
# Load the dataset
df = pd.read_csv('data/weatherAUS.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print("\nDataset info:")
print(df.info())

print("\nFirst few rows:")
print(df.head())

print("\nSummary statistics:")
print(df.describe())

print("\nClass distribution (RainTomorrow):")
print(df['RainTomorrow'].value_counts())
print(f"Class distribution percentage: {df['RainTomorrow'].value_counts(normalize=True) * 100}")

print("\nMissing values per column:")
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percent})
print(missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# Visualize missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Value Heatmap')
plt.tight_layout()
plt.savefig('missing_values_heatmap.png')

# Analyze target variable
plt.figure(figsize=(8, 5))
sns.countplot(x='RainTomorrow', data=df)
plt.title('Class Distribution: Rain Tomorrow')
plt.tight_layout()
plt.savefig('target_distribution.png')

print("\n========== 2. Data Preprocessing ==========")
# Preprocessing Function
def preprocess_data(data, mode='basic'):
    # Make a copy to avoid warnings
    data = data.copy()
    
    # Convert target to binary (Yes=1, No=0)
    le = LabelEncoder()
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'].fillna('No'))
    data['RainToday'] = le.fit_transform(data['RainToday'].fillna('No'))
    
    # Handle Date
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['Season'] = (data['Month'] % 12 + 3) // 3
    
    # Drop the Date column
    data = data.drop('Date', axis=1)
    
    # Split features and target
    X = data.drop('RainTomorrow', axis=1)
    y = data['RainTomorrow']
    
    # Advanced preprocessing if specified
    if mode == 'advanced':
        # Create feature interactions
        X['Temp_Diff'] = X['MaxTemp'] - X['MinTemp']
        X['Pressure_Diff'] = X['Pressure9am'] - X['Pressure3pm']
        X['Humidity_Diff'] = X['Humidity9am'] - X['Humidity3pm']
        X['Wind_Diff'] = X['WindSpeed3pm'] - X['WindSpeed9am']
        X['Temp_Humidity9am'] = X['Temp9am'] * X['Humidity9am']
        X['Temp_Humidity3pm'] = X['Temp3pm'] * X['Humidity3pm']
        
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    return X, y, numeric_features, categorical_features

print("Preprocessing the data...")
X, y, numeric_features, categorical_features = preprocess_data(df, mode='advanced')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, Testing set shape: {X_test.shape}")

print("\n========== 3. Building ML Pipeline ==========")

# Define preprocessing steps
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define evaluation function
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    
    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"5-Fold CV F1 Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob
    }

print("\n========== 4. Ramp-Up Approach - Training Multiple Models ==========")

# Model 1: Logistic Regression (Baseline)
print("\n--- Model 1: Logistic Regression (Baseline) ---")
log_reg_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])
log_reg_results = evaluate_model(log_reg_pipeline, X_train, X_test, y_train, y_test)

# Model 2: Random Forest 
print("\n--- Model 2: Random Forest ---")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_results = evaluate_model(rf_pipeline, X_train, X_test, y_train, y_test)

# Model 3: Gradient Boosting
print("\n--- Model 3: Gradient Boosting ---")
gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])
gb_results = evaluate_model(gb_pipeline, X_train, X_test, y_train, y_test)

# Model 4: XGBoost
print("\n--- Model 4: XGBoost ---")
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(n_estimators=100, random_state=42))
])
xgb_results = evaluate_model(xgb_pipeline, X_train, X_test, y_train, y_test)

print("\n========== 5. Model Comparison ==========")
# Collect all results
all_results = [log_reg_results, rf_results, gb_results, xgb_results]
model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'XGBoost']

# Create comparison DataFrame
comparison_df = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [res['accuracy'] for res in all_results],
    'Precision': [res['precision'] for res in all_results],
    'Recall': [res['recall'] for res in all_results],
    'F1 Score': [res['f1'] for res in all_results],
    'ROC AUC': [res['roc_auc'] for res in all_results],
    'CV F1 (5-Fold)': [res['cv_f1_mean'] for res in all_results]
})

print("Model Comparison:")
print(comparison_df.sort_values('F1 Score', ascending=False))

# Visualize results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
comparison_df_plot = comparison_df.set_index('Model')

plt.figure(figsize=(12, 6))
comparison_df_plot[metrics].plot(kind='bar', figsize=(12, 6))
plt.title('Model Performance Comparison')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('model_comparison.png')

print("\n========== 6. Hyperparameter Tuning for the Best Model ==========")

# Identify the best model based on F1 score
best_model_index = comparison_df['F1 Score'].idxmax()
best_model_name = comparison_df.loc[best_model_index, 'Model']
print(f"Best model based on F1 Score: {best_model_name}")

# Define hyperparameter grid for the best model
if best_model_name == 'Logistic Regression':
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__class_weight': [None, 'balanced']
    }
    base_model = LogisticRegression(random_state=42, max_iter=1000)
elif best_model_name == 'Random Forest':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': [None, 'balanced']
    }
    base_model = RandomForestClassifier(random_state=42)
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0]
    }
    base_model = GradientBoostingClassifier(random_state=42)
else:  # XGBoost
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__subsample': [0.8, 1.0],
        'classifier__colsample_bytree': [0.8, 1.0]
    }
    base_model = XGBClassifier(random_state=42)

# Create a new pipeline with the best model
best_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', base_model)
])

# Set up GridSearchCV
print(f"Tuning hyperparameters for {best_model_name}...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(best_pipeline, param_grid, cv=cv, scoring='f1', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get best parameters and results
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

# Evaluate the tuned model
print("\nEvaluating tuned model:")
tuned_model = grid_search.best_estimator_
tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test)

# Compare with untuned version
print("\nComparison with untuned model:")
improvement = tuned_results['f1'] - all_results[best_model_index]['f1']
print(f"F1 Score improvement: {improvement:.4f} ({improvement/all_results[best_model_index]['f1']*100:.2f}%)")

print("\n========== 7. Feature Importance Analysis ==========")

# Get feature names after preprocessing
if hasattr(preprocessor, 'transformers_'):
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend([(f"{col}") for col in cols])
        elif name == 'cat':
            # For categorical features, we need to get the encoded feature names
            if hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                cat_features = trans.named_steps['onehot'].get_feature_names_out(cols)
                feature_names.extend(cat_features)
            else:
                # Fallback for older scikit-learn versions
                feature_names.extend([f"{col}_encoded" for col in cols])

    # Get feature importances if available
    if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        feature_imp = None
        if best_model_name == 'Random Forest':
            if hasattr(tuned_model.named_steps['classifier'], 'feature_importances_'):
                feature_imp = tuned_model.named_steps['classifier'].feature_importances_
        elif best_model_name == 'Gradient Boosting':
            if hasattr(tuned_model.named_steps['classifier'], 'feature_importances_'):
                feature_imp = tuned_model.named_steps['classifier'].feature_importances_
        elif best_model_name == 'XGBoost':
            if hasattr(tuned_model.named_steps['classifier'], 'feature_importances_'):
                feature_imp = tuned_model.named_steps['classifier'].feature_importances_
        
        if feature_imp is not None and len(feature_names) == len(feature_imp):
            # Create and plot feature importance DataFrame
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_imp
            }).sort_values('Importance', ascending=False)
            
            print("\nTop 20 most important features:")
            print(feature_importance.head(20))
            
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title(f'Top 20 Feature Importances ({best_model_name})')
            plt.tight_layout()
            plt.savefig('feature_importance.png')
    elif best_model_name == 'Logistic Regression':
        # For logistic regression, extract coefficients
        if hasattr(tuned_model.named_steps['classifier'], 'coef_'):
            coefficients = tuned_model.named_steps['classifier'].coef_[0]
            if len(feature_names) == len(coefficients):
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Coefficient': coefficients
                })
                feature_importance['AbsCoefficient'] = abs(feature_importance['Coefficient'])
                feature_importance = feature_importance.sort_values('AbsCoefficient', ascending=False)
                
                print("\nTop 20 features by coefficient magnitude:")
                print(feature_importance[['Feature', 'Coefficient']].head(20))
                
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Coefficient', y='Feature', data=feature_importance.head(20))
                plt.title('Top 20 Feature Coefficients (Logistic Regression)')
                plt.tight_layout()
                plt.savefig('feature_coefficients.png')

print("\n========== 8. Saving the Best Model ==========")
# In a production setting, you would save the model here
# For example, using joblib:
# import joblib
# joblib.dump(tuned_model, 'weather_prediction_model.pkl')
print("Model training and evaluation complete.")

# Function to make predictions with the final model
def predict_rain_tomorrow(data, model):
    """
    Make predictions using the trained model
    
    Parameters:
    data (pd.DataFrame): The input data with the same columns as the training data
    model: The trained model pipeline
    
    Returns:
    np.array: Predicted probabilities of rain tomorrow
    """
    # Preprocess the data
    # Note: This assumes the data has the same structure as during training
    
    # Make predictions
    pred_proba = model.predict_proba(data)[:, 1]
    predictions = model.predict(data)
    
    return pred_proba, predictions

print("\n========== 9. Example Prediction ==========")
# Get a sample from the test data
sample_size = min(5, X_test.shape[0])
sample_data = X_test.iloc[:sample_size]
sample_actual = y_test.iloc[:sample_size]

# Make predictions
sample_proba, sample_pred = predict_rain_tomorrow(sample_data, tuned_model)

# Display results
results_df = pd.DataFrame({
    'Actual': sample_actual,
    'Predicted': sample_pred,
    'Probability': sample_proba
})

print("Sample predictions:")
print(results_df)

print("\n========== 10. Conclusion ==========")
print(f"The best model for predicting rain tomorrow is {best_model_name}.")
print(f"It achieved an F1 score of {tuned_results['f1']:.4f} on the test data.")
print("This model can be used to predict whether it will rain tomorrow based on weather conditions.")