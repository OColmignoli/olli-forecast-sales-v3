"""
Base model class for sales forecasting.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import mlflow
import logging

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, **kwargs):
        """Initialize model."""
        self.model = None
        self.params = kwargs
        self.is_trained = False
    
    @abstractmethod
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Any:
        """Preprocess data for model."""
        pass
    
    @abstractmethod
    def create_model(self) -> Any:
        """Create and configure model."""
        pass
    
    @abstractmethod
    def train_model(
        self,
        train_data: Any,
        validation_data: Optional[Any] = None
    ) -> Dict[str, float]:
        """Train model and return metrics."""
        pass
    
    @abstractmethod
    def predict_model(
        self,
        data: Any,
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        pass
    
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train model pipeline."""
        try:
            # Log parameters
            mlflow.log_params(self.params)
            
            # Preprocess data
            processed_train = self.preprocess_data(train_data, is_training=True)
            processed_val = None
            if validation_data is not None:
                processed_val = self.preprocess_data(validation_data, is_training=False)
            
            # Create model if not exists
            if self.model is None:
                self.model = self.create_model()
            
            # Train model
            metrics = self.train_model(processed_train, processed_val)
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            self.is_trained = True
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict(
        self,
        data: pd.DataFrame,
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before prediction")
            
            # Preprocess data
            processed_data = self.preprocess_data(data, is_training=False)
            
            # Generate predictions
            predictions = self.predict_model(processed_data, horizon)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
    
    def save(self, path: str) -> None:
        """Save model to path."""
        try:
            mlflow.pytorch.save_model(self.model, path)
            logger.info(f"Saved model to {path}")
        
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load(self, path: str) -> None:
        """Load model from path."""
        try:
            self.model = mlflow.pytorch.load_model(path)
            self.is_trained = True
            logger.info(f"Loaded model from {path}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
