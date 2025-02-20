"""Data preprocessing utilities for sales forecasting."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

def validate_columns(df: pd.DataFrame) -> bool:
    """Validate that required columns are present in the dataframe."""
    required_columns = [
        'Transaction Year',
        'Transaction Week',
        'Volume',
        'CV Gross Sales',
        'CV Net Sales',
        'CV COGS',
        'CV Gross Profit'
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False
    
    return True

def validate_data_types(df: pd.DataFrame) -> bool:
    """Validate data types of columns."""
    type_checks = {
        'Transaction Year': np.integer,
        'Transaction Week': np.integer,
        'Volume': np.floating,
        'CV Gross Sales': np.floating,
        'CV Net Sales': np.floating,
        'CV COGS': np.floating,
        'CV Gross Profit': np.floating
    }
    
    for col, dtype in type_checks.items():
        if not np.issubdtype(df[col].dtype, dtype):
            logger.error(f"Column {col} has incorrect type. Expected {dtype}, got {df[col].dtype}")
            return False
    
    return True

def validate_value_ranges(df: pd.DataFrame) -> bool:
    """Validate value ranges in the data."""
    range_checks = {
        'Transaction Year': (2000, 2100),  # Reasonable year range
        'Transaction Week': (1, 53),       # Valid week numbers
        'Volume': (0, None),              # Non-negative values
        'CV Gross Sales': (0, None),
        'CV Net Sales': (None, None),     # Can be negative
        'CV COGS': (0, None),
        'CV Gross Profit': (None, None)   # Can be negative
    }
    
    for col, (min_val, max_val) in range_checks.items():
        if min_val is not None and df[col].min() < min_val:
            logger.error(f"Column {col} has values below minimum {min_val}")
            return False
        if max_val is not None and df[col].max() > max_val:
            logger.error(f"Column {col} has values above maximum {max_val}")
            return False
    
    return True

def check_missing_values(df: pd.DataFrame) -> Tuple[bool, Dict[str, int]]:
    """Check for missing values in the data."""
    missing_counts = df.isnull().sum().to_dict()
    has_missing = any(missing_counts.values())
    
    if has_missing:
        logger.warning(f"Found missing values: {missing_counts}")
    
    return not has_missing, missing_counts

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create date-related features from Transaction Year and Week."""
    df = df.copy()
    
    # Create date column
    df['date'] = df.apply(
        lambda row: pd.to_datetime(f"{row['Transaction Year']}-W{row['Transaction Week']:02d}-1", format='%Y-W%W-%w'),
        axis=1
    )
    
    # Extract useful date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week
    
    return df

def create_lag_features(df: pd.DataFrame, columns: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lag features for specified columns."""
    df = df.copy()
    
    for col in columns:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
    return df

def create_rolling_features(df: pd.DataFrame, columns: List[str], windows: List[int]) -> pd.DataFrame:
    """Create rolling window features for specified columns."""
    df = df.copy()
    
    for col in columns:
        for window in windows:
            df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()
            df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window, min_periods=1).min()
    
    return df

def handle_missing_values(df: pd.DataFrame, strategy: str = 'interpolate') -> pd.DataFrame:
    """Handle missing values in the data."""
    df = df.copy()
    
    if strategy == 'interpolate':
        df = df.interpolate(method='linear')
    elif strategy == 'forward_fill':
        df = df.fillna(method='ffill')
    elif strategy == 'backward_fill':
        df = df.fillna(method='bfill')
    elif strategy == 'mean':
        df = df.fillna(df.mean())
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Fill any remaining NaNs with 0
    df = df.fillna(0)
    
    return df

def normalize_features(df: pd.DataFrame, columns: List[str]) -> Tuple[pd.DataFrame, Dict[str, MinMaxScaler]]:
    """Normalize specified features using MinMaxScaler."""
    df = df.copy()
    scalers = {}
    
    for col in columns:
        scaler = MinMaxScaler()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    
    return df, scalers

def prepare_data_for_training(
    df: pd.DataFrame,
    target_column: str,
    sequence_length: int = 52,
    forecast_horizon: int = 13,
    feature_columns: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data for model training."""
    if feature_columns is None:
        feature_columns = [
            'Volume',
            'CV Gross Sales',
            'CV Net Sales',
            'CV COGS',
            'CV Gross Profit'
        ]
    
    # Validate data
    if not validate_columns(df):
        raise ValueError("Data validation failed: missing columns")
    
    if not validate_data_types(df):
        raise ValueError("Data validation failed: incorrect data types")
    
    if not validate_value_ranges(df):
        raise ValueError("Data validation failed: invalid value ranges")
    
    # Create features
    df = create_date_features(df)
    df = create_lag_features(df, [target_column], [1, 2, 3, 4, 8, 13, 26, 52])
    df = create_rolling_features(df, [target_column], [4, 8, 13, 26, 52])
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Normalize features
    df, _ = normalize_features(df, feature_columns)
    
    # Create sequences
    X, y = create_sequences(df[feature_columns].values, df[target_column].values,
                          sequence_length, forecast_horizon)
    
    return X, y

def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    sequence_length: int,
    forecast_horizon: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series forecasting."""
    X, y = [], []
    
    for i in range(len(features) - sequence_length - forecast_horizon + 1):
        X.append(features[i:(i + sequence_length)])
        y.append(target[i + sequence_length:i + sequence_length + forecast_horizon])
    
    return np.array(X), np.array(y)
