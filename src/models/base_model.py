"""Base model class for all forecasting models."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from .config import ModelConfig

class BaseModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, config: ModelConfig, workspace_name: str = "OLLI_ML_Forecast",
                 resource_group: str = "OLLI-resource", subscription_id: str = "c828c783-7a28-48f4-b56f-a6c189437d77"):
        """Initialize the model with configuration and Azure ML workspace."""
        self.config = config
        self.model = None
        self.scaler = None
        self.ml_client = MLClient(
            DefaultAzureCredential(),
            subscription_id,
            resource_group,
            workspace_name
        )
        
    @abstractmethod
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess the input data for model training."""
        pass
    
    @abstractmethod
    def build_model(self) -> Any:
        """Build and return the model architecture."""
        pass
    
    @abstractmethod
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the model and return training history."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame, horizon: int = 13) -> np.ndarray:
        """Generate predictions for the given horizon."""
        pass
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save the model to Azure ML workspace."""
        if model_name is None:
            model_name = f"{self.config.name}_model"
        
        # Implementation for saving model to Azure ML
        # This will be implemented specifically for each model type
        pass
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load the model from Azure ML workspace."""
        if model_name is None:
            model_name = f"{self.config.name}_model"
        
        # Implementation for loading model from Azure ML
        # This will be implemented specifically for each model type
        pass
    
    def create_sequences(self, data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series data."""
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:(i + seq_length)])
            y.append(data[i + seq_length])
        return np.array(X), np.array(y)
    
    def evaluate(self, true_values: np.ndarray, predictions: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance using various metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        mse = mean_squared_error(true_values, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true_values, predictions)
        mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
