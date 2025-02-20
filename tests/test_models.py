"""Test suite for forecasting models."""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.deepar_model import DeepARModel
from src.models.cnn_model import CNNModel
from src.models.prophet_model import ProphetModel
from src.models.stacking_model import StackingModel
from src.models.config import (
    LSTMConfig,
    TransformerConfig,
    DeepARConfig,
    CNNConfig,
    ProphetConfig,
    StackingConfig
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

@pytest.fixture
def mock_ml_client():
    """Create mock Azure ML client."""
    return Mock()

def test_lstm_model(sample_data, mock_ml_client):
    """Test LSTM model."""
    config = LSTMConfig(
        hidden_units=[64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
    
    model = LSTMModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == len(config.output_features)

def test_transformer_model(sample_data, mock_ml_client):
    """Test Transformer model."""
    config = TransformerConfig(
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=128
    )
    
    model = TransformerModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == len(config.output_features)

def test_deepar_model(sample_data, mock_ml_client):
    """Test DeepAR+ model."""
    config = DeepARConfig(
        hidden_size=40,
        num_layers=2,
        dropout_rate=0.1,
        learning_rate=0.001
    )
    
    model = DeepARModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, dict)
    assert 'mean' in predictions
    assert 'std' in predictions
    assert isinstance(predictions['mean'], np.ndarray)
    assert predictions['mean'].shape[1] == len(config.output_features)

def test_cnn_model(sample_data, mock_ml_client):
    """Test CNN model."""
    config = CNNConfig(
        filters=[32, 64, 128],
        kernel_sizes=[3, 3, 3],
        dense_units=[64, 32],
        dropout_rate=0.2
    )
    
    model = CNNModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == len(config.output_features)

def test_prophet_model(sample_data, mock_ml_client):
    """Test Prophet model."""
    config = ProphetConfig(
        seasonality_mode='multiplicative',
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    model = ProphetModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mse' in metrics
    assert 'mae' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == len(config.output_features)

def test_stacking_model(sample_data, mock_ml_client):
    """Test Stacking model."""
    config = StackingConfig(
        base_models=['lstm', 'transformer', 'deepar', 'cnn', 'prophet'],
        meta_model='xgboost'
    )
    
    model = StackingModel(config)
    model.ml_client = mock_ml_client
    
    # Test training
    metrics = model.train(sample_data)
    assert isinstance(metrics, dict)
    assert 'mean_cv_score' in metrics
    assert 'std_cv_score' in metrics
    
    # Test prediction
    predictions = model.predict(sample_data)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions.shape) == 2
    assert predictions.shape[1] == len(config.output_features)
