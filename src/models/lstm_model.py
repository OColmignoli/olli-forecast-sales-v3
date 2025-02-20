"""LSTM model implementation for time series forecasting."""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import mlflow.tensorflow

from .base_model import BaseModel
from .config import LSTMConfig

class LSTMModel(BaseModel):
    """LSTM implementation using TensorFlow/Keras."""
    
    def __init__(self, config: LSTMConfig):
        """Initialize LSTM model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for LSTM model."""
        # Extract features and target
        features = data[self.config.input_features].values
        target = data[self.config.target_feature].values
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.config.sequence_length)
        
        return X, y
    
    def build_model(self) -> tf.keras.Model:
        """Build LSTM model architecture."""
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(
                self.config.hidden_units[0],
                input_shape=(self.config.sequence_length, len(self.config.input_features)),
                return_sequences=True
            ),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.LSTM(self.config.hidden_units[1]),
            tf.keras.layers.Dropout(self.config.dropout_rate),
            tf.keras.layers.Dense(len(self.config.input_features))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the LSTM model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "hidden_units": self.config.hidden_units,
            "dropout_rate": self.config.dropout_rate,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "epochs": self.config.epochs
        })
        
        # Preprocess training data
        X_train, y_train = self.preprocess_data(data)
        
        # Preprocess validation data if provided
        if validation_data is not None:
            X_val, y_val = self.preprocess_data(validation_data)
            validation_data = (X_val, y_val)
        
        # Build and train model
        self.model = self.build_model()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log metrics
        mlflow.log_metrics({
            "final_loss": history.history['loss'][-1],
            "final_mae": history.history['mae'][-1]
        })
        
        # Log model
        mlflow.tensorflow.log_model(self.model, "model")
        
        # End MLflow run
        mlflow.end_run()
        
        return history.history
    
    def predict(self, data: pd.DataFrame, horizon: int = 13) -> np.ndarray:
        """Generate predictions using the trained LSTM model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess input data
        features_scaled = self.scaler.transform(data[self.config.input_features].values)
        
        # Generate predictions
        predictions = []
        current_sequence = features_scaled[-self.config.sequence_length:]
        
        for _ in range(horizon):
            # Reshape sequence for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.config.sequence_length, -1)
            
            # Get next prediction
            next_pred = self.model.predict(current_sequence_reshaped, verbose=0)
            predictions.append(next_pred[0])
            
            # Update sequence
            current_sequence = np.vstack((current_sequence[1:], next_pred))
        
        # Inverse transform predictions
        predictions = np.array(predictions)
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        
        return predictions_rescaled
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save LSTM model to Azure ML workspace."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if model_name is None:
            model_name = f"{self.config.name}_model"
            
        # Save model locally first
        local_path = f"./tmp/{model_name}"
        self.model.save(local_path)
        
        # Register model in Azure ML workspace
        model = Model(
            path=local_path,
            name=model_name,
            description="LSTM model for sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load LSTM model from Azure ML workspace."""
        if model_name is None:
            model_name = f"{self.config.name}_model"
            
        # Get model from Azure ML workspace
        model = self.ml_client.models.get(name=model_name, label="latest")
        
        # Download model
        self.ml_client.models.download(
            name=model_name,
            version=model.version,
            download_path="./tmp"
        )
        
        # Load the model
        self.model = tf.keras.models.load_model(f"./tmp/{model_name}")
