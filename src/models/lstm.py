"""
LSTM model for sales forecasting.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import mlflow.tensorflow
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

class LSTMModel(BaseModel):
    """LSTM model for sales forecasting."""
    
    def __init__(
        self,
        sequence_length: int = 52,
        hidden_units: int = 64,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        """Initialize LSTM model."""
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set random seed
        tf.random.set_seed(42)
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM model."""
        try:
            # Extract features and target
            features = data[[
                'Transaction Year',
                'Transaction Week',
                'Volume',
                'CV Gross Sales',
                'CV Net Sales',
                'CV COGS'
            ]].values
            
            target = data['CV Gross Profit'].values
            
            # Scale data
            if is_training:
                self.feature_scaler = tf.keras.preprocessing.StandardScaler()
                self.target_scaler = tf.keras.preprocessing.StandardScaler()
                
                scaled_features = self.feature_scaler.fit_transform(features)
                scaled_target = self.target_scaler.fit_transform(target.reshape(-1, 1))
            else:
                scaled_features = self.feature_scaler.transform(features)
                scaled_target = self.target_scaler.transform(target.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - self.sequence_length):
                X.append(scaled_features[i:i + self.sequence_length])
                y.append(scaled_target[i + self.sequence_length])
            
            return np.array(X), np.array(y)
        
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def create_model(self) -> Sequential:
        """Create LSTM model."""
        try:
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(
                units=self.hidden_units,
                return_sequences=True if self.num_layers > 1 else False,
                input_shape=(self.sequence_length, 6)
            ))
            model.add(Dropout(self.dropout_rate))
            
            # Additional LSTM layers
            for i in range(self.num_layers - 1):
                model.add(LSTM(
                    units=self.hidden_units,
                    return_sequences=True if i < self.num_layers - 2 else False
                ))
                model.add(Dropout(self.dropout_rate))
            
            # Output layer
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=self.learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Train LSTM model."""
        try:
            X_train, y_train = train_data
            
            # Configure callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if validation_data else 'loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            )
            
            # Get metrics
            metrics = {
                'loss': history.history['loss'][-1],
                'mae': history.history['mae'][-1]
            }
            
            if validation_data:
                metrics.update({
                    'val_loss': history.history['val_loss'][-1],
                    'val_mae': history.history['val_mae'][-1]
                })
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict_model(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            X, _ = data
            
            # Generate predictions iteratively
            predictions = []
            current_sequence = X[-1]
            
            for _ in range(horizon):
                # Predict next value
                pred = self.model.predict(
                    current_sequence.reshape(1, self.sequence_length, 6),
                    verbose=0
                )
                predictions.append(pred[0, 0])
                
                # Update sequence
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1, 2:] = pred  # Update features with prediction
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = self.target_scaler.inverse_transform(predictions)
            
            return predictions.flatten()
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
