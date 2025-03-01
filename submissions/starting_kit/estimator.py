import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformers
class TemporalFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract temporal features from date column."""
    
    def __init__(self):
        self.date_column = 'Date'
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if self.date_column in X_copy.columns:
            # Convert to datetime
            X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column])
            
            # Extract date components
            X_copy['year'] = X_copy[self.date_column].dt.year
            X_copy['month'] = X_copy[self.date_column].dt.month
            X_copy['day'] = X_copy[self.date_column].dt.day
            X_copy['dayofweek'] = X_copy[self.date_column].dt.dayofweek
            
            # Calculate season (Southern Hemisphere)
            X_copy['season'] = ((X_copy['month'] % 12 + 3) // 3) % 4 + 1
            
            # Drop original date column
            X_copy = X_copy.drop(self.date_column, axis=1)
        
        return X_copy


class WeatherDifferentialExtractor(BaseEstimator, TransformerMixin):
    """Calculate differences and ratios between measurements."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Temperature differentials
        if 'MinTemp' in X_copy.columns and 'MaxTemp' in X_copy.columns:
            X_copy['temp_range'] = X_copy['MaxTemp'] - X_copy['MinTemp']
        
        if 'Temp9am' in X_copy.columns and 'Temp3pm' in X_copy.columns:
            X_copy['temp_change'] = X_copy['Temp3pm'] - X_copy['Temp9am']
        
        # Pressure differentials
        if 'Pressure9am' in X_copy.columns and 'Pressure3pm' in X_copy.columns:
            X_copy['pressure_change'] = X_copy['Pressure3pm'] - X_copy['Pressure9am']
        
        # Humidity differentials
        if 'Humidity9am' in X_copy.columns and 'Humidity3pm' in X_copy.columns:
            X_copy['humidity_change'] = X_copy['Humidity3pm'] - X_copy['Humidity9am']
        
        # Wind speed differentials
        if 'WindSpeed9am' in X_copy.columns and 'WindSpeed3pm' in X_copy.columns:
            X_copy['wind_change'] = X_copy['WindSpeed3pm'] - X_copy['WindSpeed9am']
        
        # Cloud cover differentials
        if 'Cloud9am' in X_copy.columns and 'Cloud3pm' in X_copy.columns:
            X_copy['cloud_change'] = X_copy['Cloud3pm'] - X_copy['Cloud9am']
        
        # Weather interaction features
        if 'Temp3pm' in X_copy.columns and 'Humidity3pm' in X_copy.columns:
            X_copy['temp_humidity_product'] = X_copy['Temp3pm'] * X_copy['Humidity3pm']
        
        # Pressure-wind interactions
        if 'Pressure3pm' in X_copy.columns and 'WindSpeed3pm' in X_copy.columns:
            X_copy['pressure_wind_ratio'] = X_copy['Pressure3pm'] / (X_copy['WindSpeed3pm'] + 0.1)
        
        # Pressure gradient and temperature
        if 'pressure_change' in X_copy.columns and 'temp_change' in X_copy.columns:
            X_copy['pressure_temp_ratio'] = X_copy['pressure_change'] / (abs(X_copy['temp_change']) + 0.1)
        
        return X_copy


class CyclicalFeatureEncoder(BaseEstimator, TransformerMixin):
    """Encode cyclical features like month, day, and wind direction using sine/cosine transformation."""
    
    def __init__(self):
        self.cyclical_features = ['month', 'day', 'dayofweek']
        self.wind_direction_cols = ['WindGustDir', 'WindDir9am', 'WindDir3pm']
        self.direction_mapping = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }
        self.max_values = {'month': 12, 'day': 31, 'dayofweek': 7}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # Transform cyclical numerical features
        for col in self.cyclical_features:
            if col in X_copy.columns:
                max_val = self.max_values.get(col, 1)
                X_copy[f'{col}_sin'] = np.sin(2 * np.pi * X_copy[col] / max_val)
                X_copy[f'{col}_cos'] = np.cos(2 * np.pi * X_copy[col] / max_val)
                X_copy = X_copy.drop(col, axis=1)
        
        # Transform wind direction columns
        for col in self.wind_direction_cols:
            if col in X_copy.columns:
                # Convert directions to angles, with NaN/missing preserved
                angles = X_copy[col].map(self.direction_mapping)
                
                # Convert to radians and calculate sin/cos components
                X_copy[f'{col}_sin'] = np.sin(np.radians(angles))
                X_copy[f'{col}_cos'] = np.cos(np.radians(angles))
                
                # Drop original column
                X_copy = X_copy.drop(col, axis=1)
        
        return X_copy


class RainTodayEncoder(BaseEstimator, TransformerMixin):
    """Encode RainToday as numeric and add derived features."""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if 'RainToday' in X_copy.columns:
            # Map Yes/No to 1/0
            rain_map = {'Yes': 1, 'No': 0}
            X_copy['RainToday_numeric'] = X_copy['RainToday'].map(rain_map)
            
            # Create interaction features with RainToday
            if 'Humidity3pm' in X_copy.columns:
                X_copy['Rain_Humidity'] = X_copy['RainToday_numeric'] * X_copy['Humidity3pm']
                
            if 'Pressure3pm' in X_copy.columns:
                X_copy['Rain_Pressure'] = X_copy['RainToday_numeric'] * X_copy['Pressure3pm']
                
            # Drop original column
            X_copy = X_copy.drop('RainToday', axis=1)
        
        return X_copy


class LocationEncoder(BaseEstimator, TransformerMixin):
    """Handle Location feature using various statistical aggregations."""
    
    def __init__(self):
        self.location_stats = {}
    
    def fit(self, X, y=None):
        if 'Location' in X.columns and y is not None:
            # Calculate rain probability by location
            location_stats = pd.DataFrame({
                'rain_prob': X.groupby('Location')['RainToday_numeric'].mean(),
                'temp_mean': X.groupby('Location')['MaxTemp'].mean(),
                'humidity_mean': X.groupby('Location')['Humidity3pm'].mean() if 'Humidity3pm' in X.columns else None,
                'pressure_mean': X.groupby('Location')['Pressure3pm'].mean() if 'Pressure3pm' in X.columns else None
            })
            
            # Store for transform
            self.location_stats = location_stats.to_dict(orient='index')
        
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        if 'Location' in X_copy.columns and self.location_stats:
            # Add location-based features
            X_copy['location_rain_prob'] = X_copy['Location'].map(
                {loc: stats.get('rain_prob', 0) for loc, stats in self.location_stats.items()}
            ).fillna(0)
            
            X_copy['location_temp_mean'] = X_copy['Location'].map(
                {loc: stats.get('temp_mean', 0) for loc, stats in self.location_stats.items()}
            ).fillna(0)
            
            if 'humidity_mean' in next(iter(self.location_stats.values())):
                X_copy['location_humidity_mean'] = X_copy['Location'].map(
                    {loc: stats.get('humidity_mean', 0) for loc, stats in self.location_stats.items()}
                ).fillna(0)
            
            if 'pressure_mean' in next(iter(self.location_stats.values())):
                X_copy['location_pressure_mean'] = X_copy['Location'].map(
                    {loc: stats.get('pressure_mean', 0) for loc, stats in self.location_stats.items()}
                ).fillna(0)
            
            # Drop original location column
            X_copy = X_copy.drop('Location', axis=1)
        
        return X_copy


# Define feature groups for preprocessing
def identify_feature_groups(X):
    """Identify different feature groups for preprocessing."""
    
    columns = X.columns.tolist()
    
    # Categorical features
    categorical_features = ['Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RainToday'] 
    categorical_features = [col for col in categorical_features if col in columns]
    
    # Numerical features
    numerical_features = [
        'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
        'WindGustSpeed', 'WindSpeed9am', 'WindSpeed3pm', 
        'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm',
        'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm'
    ]
    numerical_features = [col for col in numerical_features if col in columns]
    
    return categorical_features, numerical_features


# Custom function to calculate weather severity index
def calculate_weather_severity(X_df):
    """Calculate a weather severity index based on multiple factors."""
    
    df = X_df.copy()
    
    # Initialize severity with zeros
    severity = np.zeros(len(df))
    
    # Add contribution from various weather factors
    if 'Rainfall' in df.columns:
        # Normalize rainfall (higher rainfall -> higher severity)
        rainfall_norm = df['Rainfall'] / df['Rainfall'].max() if df['Rainfall'].max() > 0 else 0
        severity += rainfall_norm * 0.3  # 30% weight
    
    if 'WindGustSpeed' in df.columns:
        # Normalize wind speed (higher wind -> higher severity)
        wind_norm = df['WindGustSpeed'] / df['WindGustSpeed'].max() if df['WindGustSpeed'].max() > 0 else 0
        severity += wind_norm * 0.2  # 20% weight
    
    if 'Humidity3pm' in df.columns:
        # Normalize humidity (higher humidity -> higher severity)
        humidity_norm = df['Humidity3pm'] / 100  # Already on a 0-100 scale
        severity += humidity_norm * 0.15  # 15% weight
    
    if 'Pressure3pm' in df.columns:
        # Normalize pressure (lower pressure -> higher severity)
        # Invert pressure scale since lower pressure often correlates with bad weather
        pressure_min = df['Pressure3pm'].min() if df['Pressure3pm'].min() > 0 else 990
        pressure_max = df['Pressure3pm'].max() if df['Pressure3pm'].max() > 0 else 1030
        pressure_range = pressure_max - pressure_min
        if pressure_range > 0:
            pressure_norm = 1 - ((df['Pressure3pm'] - pressure_min) / pressure_range)
            severity += pressure_norm * 0.15  # 15% weight
    
    if 'temp_range' in df.columns:
        # Normalize temperature range (higher range -> higher severity)
        temp_range_norm = df['temp_range'] / df['temp_range'].max() if df['temp_range'].max() > 0 else 0
        severity += temp_range_norm * 0.1  # 10% weight
    
    if 'Cloud3pm' in df.columns:
        # Normalize cloud cover (higher cloud cover -> higher severity)
        cloud_norm = df['Cloud3pm'] / 8  # Cloud cover is typically on a 0-8 scale
        severity += cloud_norm * 0.1  # 10% weight
    
    return severity.reshape(-1, 1)  # Return as 2D array for sklearn


# Custom transformer for weather severity
transformer_severity = FunctionTransformer(
    calculate_weather_severity
)


def get_estimator():
    """Return the model pipeline."""
    
    # Create pre-processing steps with custom transformers
    preprocessing_pipeline = Pipeline([
        ('temporal_features', TemporalFeatureExtractor()),
        ('weather_differentials', WeatherDifferentialExtractor()),
        ('rain_today_encoding', RainTodayEncoder()),
        ('location_encoding', LocationEncoder()),
        ('cyclical_encoding', CyclicalFeatureEncoder())
    ])
    
    # Function to apply custom preprocessing and identify features for final pipeline
    def custom_preprocessor(X, y=None):
        # Apply initial preprocessing
        X_processed = preprocessing_pipeline.fit_transform(X, y)
        
        # Identify feature groups in processed data
        return X_processed
    
    # Define column transformer for final preprocessing
    final_transformer = FunctionTransformer(
        custom_preprocessor
    )
    
    # Create main pipeline
    pipe = Pipeline([
        ('preprocessing', final_transformer),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=200, 
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    
    return pipe