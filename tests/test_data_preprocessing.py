"""Test suite for data preprocessing utilities."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.utils.data_preprocessing import (
    validate_columns,
    validate_data_types,
    validate_value_ranges,
    check_missing_values,
    create_date_features,
    create_lag_features,
    create_rolling_features,
    handle_missing_values,
    normalize_features,
    prepare_data_for_training
)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='W-MON')
    df = pd.DataFrame({
        'Transaction Year': dates.year,
        'Transaction Week': dates.isocalendar().week,
        'Volume': np.random.normal(1000, 100, 100),
        'CV Gross Sales': np.random.normal(5000, 500, 100),
        'CV Net Sales': np.random.normal(4000, 400, 100),
        'CV COGS': np.random.normal(3000, 300, 100),
        'CV Gross Profit': np.random.normal(1000, 100, 100)
    })
    return df

def test_validate_columns(sample_data):
    """Test column validation."""
    # Test valid data
    assert validate_columns(sample_data) is True
    
    # Test missing columns
    invalid_data = sample_data.drop('Volume', axis=1)
    assert validate_columns(invalid_data) is False
    
    # Test extra columns
    invalid_data = sample_data.copy()
    invalid_data['Extra'] = 0
    assert validate_columns(invalid_data) is True

def test_validate_data_types(sample_data):
    """Test data type validation."""
    # Test valid data
    assert validate_data_types(sample_data) is True
    
    # Test invalid types
    invalid_data = sample_data.copy()
    invalid_data['Volume'] = invalid_data['Volume'].astype(str)
    assert validate_data_types(invalid_data) is False

def test_validate_value_ranges(sample_data):
    """Test value range validation."""
    # Test valid data
    assert validate_value_ranges(sample_data) is True
    
    # Test invalid week numbers
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Transaction Week'] = 54
    assert validate_value_ranges(invalid_data) is False
    
    # Test invalid year
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Transaction Year'] = 1900
    assert validate_value_ranges(invalid_data) is False
    
    # Test negative values
    invalid_data = sample_data.copy()
    invalid_data.loc[0, 'Volume'] = -1
    assert validate_value_ranges(invalid_data) is False

def test_check_missing_values(sample_data):
    """Test missing value detection."""
    # Test complete data
    has_no_missing, missing_counts = check_missing_values(sample_data)
    assert has_no_missing is True
    assert all(count == 0 for count in missing_counts.values())
    
    # Test data with missing values
    incomplete_data = sample_data.copy()
    incomplete_data.loc[0, 'Volume'] = np.nan
    has_missing, missing_counts = check_missing_values(incomplete_data)
    assert has_missing is False
    assert missing_counts['Volume'] == 1

def test_create_date_features(sample_data):
    """Test date feature creation."""
    result = create_date_features(sample_data)
    
    # Check new columns
    assert 'date' in result.columns
    assert 'year' in result.columns
    assert 'month' in result.columns
    assert 'quarter' in result.columns
    assert 'day_of_week' in result.columns
    assert 'week_of_year' in result.columns
    
    # Check values
    first_row = result.iloc[0]
    assert first_row['year'] == 2020
    assert 1 <= first_row['month'] <= 12
    assert 1 <= first_row['quarter'] <= 4
    assert 0 <= first_row['day_of_week'] <= 6
    assert 1 <= first_row['week_of_year'] <= 53

def test_create_lag_features(sample_data):
    """Test lag feature creation."""
    columns = ['Volume']
    lags = [1, 2]
    result = create_lag_features(sample_data, columns, lags)
    
    # Check new columns
    for col in columns:
        for lag in lags:
            assert f'{col}_lag_{lag}' in result.columns
    
    # Check values
    assert result['Volume_lag_1'].iloc[1] == sample_data['Volume'].iloc[0]
    assert result['Volume_lag_2'].iloc[2] == sample_data['Volume'].iloc[0]

def test_create_rolling_features(sample_data):
    """Test rolling feature creation."""
    columns = ['Volume']
    windows = [2, 3]
    result = create_rolling_features(sample_data, columns, windows)
    
    # Check new columns
    for col in columns:
        for window in windows:
            assert f'{col}_rolling_mean_{window}' in result.columns
            assert f'{col}_rolling_std_{window}' in result.columns
            assert f'{col}_rolling_max_{window}' in result.columns
            assert f'{col}_rolling_min_{window}' in result.columns

def test_handle_missing_values(sample_data):
    """Test missing value handling."""
    # Create data with missing values
    data = sample_data.copy()
    data.loc[0, 'Volume'] = np.nan
    data.loc[1, 'Volume'] = np.nan
    
    # Test interpolation
    result = handle_missing_values(data, strategy='interpolate')
    assert not result['Volume'].isna().any()
    
    # Test forward fill
    result = handle_missing_values(data, strategy='forward_fill')
    assert not result['Volume'].isna().any()
    
    # Test backward fill
    result = handle_missing_values(data, strategy='backward_fill')
    assert not result['Volume'].isna().any()
    
    # Test mean fill
    result = handle_missing_values(data, strategy='mean')
    assert not result['Volume'].isna().any()
    
    # Test invalid strategy
    with pytest.raises(ValueError):
        handle_missing_values(data, strategy='invalid')

def test_normalize_features(sample_data):
    """Test feature normalization."""
    columns = ['Volume', 'CV Net Sales']
    result, scalers = normalize_features(sample_data, columns)
    
    # Check normalized values
    for col in columns:
        assert result[col].min() >= 0
        assert result[col].max() <= 1
        
    # Check scalers
    assert set(scalers.keys()) == set(columns)
    for col in columns:
        assert scalers[col] is not None

def test_prepare_data_for_training(sample_data):
    """Test data preparation for training."""
    X, y = prepare_data_for_training(
        sample_data,
        target_column='CV Net Sales',
        sequence_length=4,
        forecast_horizon=2
    )
    
    # Check shapes
    assert len(X.shape) == 3  # (samples, sequence_length, features)
    assert len(y.shape) == 2  # (samples, forecast_horizon)
    assert X.shape[0] == y.shape[0]  # Same number of samples
    assert X.shape[1] == 4  # sequence_length
    assert y.shape[1] == 2  # forecast_horizon
