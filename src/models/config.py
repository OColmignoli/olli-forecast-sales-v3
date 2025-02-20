"""Configuration for ML models and training."""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelConfig:
    """Base configuration for all models."""
    name: str
    input_features: List[str]
    target_feature: str
    sequence_length: int = 52  # One year of weekly data
    forecast_horizon: int = 13  # Quarter ahead forecast

@dataclass
class LSTMConfig(ModelConfig):
    """Configuration for LSTM model."""
    hidden_units: List[int] = (64, 32)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

@dataclass
class TransformerConfig(ModelConfig):
    """Configuration for Temporal Fusion Transformer."""
    d_model: int = 64
    num_heads: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout_rate: float = 0.1
    learning_rate: float = 0.0001
    batch_size: int = 32
    epochs: int = 100

@dataclass
class DeepARConfig(ModelConfig):
    """Configuration for DeepAR+ model."""
    num_layers: int = 2
    hidden_size: int = 40
    context_length: int = 52
    prediction_length: int = 13
    batch_size: int = 32
    epochs: int = 100

@dataclass
class CNNConfig(ModelConfig):
    """Configuration for CNN model."""
    filters: List[int] = (32, 64, 128)
    kernel_sizes: List[int] = (3, 3, 3)
    dense_units: List[int] = (64, 32)
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100

@dataclass
class ProphetConfig(ModelConfig):
    """Configuration for Prophet model."""
    seasonality_mode: str = "multiplicative"
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0

@dataclass
class StackingConfig(ModelConfig):
    """Configuration for Stacking meta-model."""
    base_models: List[str] = ("lstm", "transformer", "deepar", "cnn", "prophet")
    meta_model: str = "xgboost"  # or "randomforest"
    cv_folds: int = 5
    use_probabilities: bool = True
