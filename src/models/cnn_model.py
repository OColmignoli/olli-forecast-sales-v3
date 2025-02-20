"""CNN model implementation for time series forecasting."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import mlflow.tensorflow

from .base_model import BaseModel
from .config import CNNConfig

class CNNModel(BaseModel):
    """CNN implementation using TensorFlow/Keras."""
    
    def __init__(self, config: CNNConfig):
        """Initialize CNN model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for CNN model."""
        # Extract features and target
        features = data[self.config.input_features].values
        target = data[self.config.target_feature].values
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.config.sequence_length)
        
        # Reshape for CNN (batch_size, sequence_length, n_features, 1)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        
        return X, y
    
    def build_model(self) -> tf.keras.Model:
        """Build CNN model architecture."""
        model = tf.keras.Sequential()
        
        # First Conv1D layer
        model.add(tf.keras.layers.Conv2D(
            filters=self.config.filters[0],
            kernel_size=(self.config.kernel_sizes[0], 1),
            activation='relu',
            input_shape=(self.config.sequence_length, len(self.config.input_features), 1)
        ))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
        
        # Additional Conv1D layers
        for filters, kernel_size in zip(self.config.filters[1:], self.config.kernel_sizes[1:]):
            model.add(tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=(kernel_size, 1),
                activation='relu'
            ))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 1)))
        
        # Flatten layer
        model.add(tf.keras.layers.Flatten())
        
        # Dense layers
        for units in self.config.dense_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
            model.add(tf.keras.layers.Dropout(self.config.dropout_rate))
            model.add(tf.keras.layers.BatchNormalization())
        
        # Output layer
        model.add(tf.keras.layers.Dense(len(self.config.input_features)))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the CNN model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "filters": self.config.filters,
            "kernel_sizes": self.config.kernel_sizes,
            "dense_units": self.config.dense_units,
            "dropout_rate": self.config.dropout_rate,
            "learning_rate": self.config.learning_rate
        })
        
        # Preprocess training data
        X_train, y_train = self.preprocess_data(data)
        
        # Preprocess validation data if provided
        validation_split = 0.2
        if validation_data is not None:
            X_val, y_val = self.preprocess_data(validation_data)
            validation_data = (X_val, y_val)
            validation_split = 0.0
        
        # Build and train model
        self.model = self.build_model()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
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
            validation_split=validation_split,
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
        """Generate predictions using the trained CNN model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Preprocess input data
        features_scaled = self.scaler.transform(data[self.config.input_features].values)
        
        # Generate predictions
        predictions = []
        current_sequence = features_scaled[-self.config.sequence_length:]
        
        for _ in range(horizon):
            # Reshape sequence for prediction
            current_sequence_reshaped = current_sequence.reshape(1, self.config.sequence_length, -1, 1)
            
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
        """Save CNN model to Azure ML workspace."""
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
            description="CNN model for sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load CNN model from Azure ML workspace."""
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
