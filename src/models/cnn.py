"""
CNN model for sales forecasting.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import mlflow.tensorflow
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

class CNNModel(BaseModel):
    """CNN model for sales forecasting."""
    
    def __init__(
        self,
        sequence_length: int = 52,
        filters: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        num_conv_layers: int = 2,
        dense_units: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        """Initialize CNN model."""
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.num_conv_layers = num_conv_layers
        self.dense_units = dense_units
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
        """Preprocess data for CNN model."""
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
        """Create CNN model."""
        try:
            model = Sequential()
            
            # First Conv1D layer
            model.add(Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                activation='relu',
                padding='same',
                input_shape=(self.sequence_length, 6)
            ))
            model.add(MaxPooling1D(pool_size=self.pool_size))
            model.add(Dropout(self.dropout_rate))
            
            # Additional Conv1D layers
            for _ in range(self.num_conv_layers - 1):
                model.add(Conv1D(
                    filters=self.filters,
                    kernel_size=self.kernel_size,
                    activation='relu',
                    padding='same'
                ))
                model.add(MaxPooling1D(pool_size=self.pool_size))
                model.add(Dropout(self.dropout_rate))
            
            # Flatten layer
            model.add(Flatten())
            
            # Dense layers
            model.add(Dense(self.dense_units, activation='relu'))
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
        """Train CNN model."""
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
    
    def _create_feature_maps(
        self,
        input_sequence: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Create feature maps for visualization."""
        try:
            # Create intermediate model to get conv layer outputs
            layer_outputs = []
            for layer in self.model.layers:
                if isinstance(layer, Conv1D):
                    layer_outputs.append(layer.output)
            
            feature_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=layer_outputs
            )
            
            # Get feature maps
            feature_maps = feature_model.predict(input_sequence.reshape(1, self.sequence_length, 6))
            
            # Create dictionary of feature maps
            feature_dict = {
                f'conv_layer_{i}': maps
                for i, maps in enumerate(feature_maps)
            }
            
            return feature_dict
        
        except Exception as e:
            logger.error(f"Failed to create feature maps: {str(e)}")
            raise
    
    def analyze_patterns(
        self,
        input_sequence: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze learned patterns in the input sequence."""
        try:
            # Get feature maps
            feature_maps = self._create_feature_maps(input_sequence)
            
            # Analyze activation patterns
            analysis = {}
            
            for layer_name, maps in feature_maps.items():
                # Calculate activation statistics
                analysis[layer_name] = {
                    'mean_activation': float(np.mean(maps)),
                    'max_activation': float(np.max(maps)),
                    'active_filters': int(np.sum(np.max(maps, axis=(1, 2)) > 0)),
                    'total_filters': maps.shape[-1]
                }
            
            return analysis
        
        except Exception as e:
            logger.error(f"Failed to analyze patterns: {str(e)}")
            raise
