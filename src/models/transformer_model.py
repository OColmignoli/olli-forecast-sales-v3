"""Transformer model implementation for time series forecasting."""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import mlflow.pytorch

from .base_model import BaseModel
from .config import TransformerConfig

class TimeSeriesTransformer(pl.LightningModule):
    """PyTorch Lightning implementation of Temporal Fusion Transformer."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize the transformer model."""
        super().__init__()
        self.config = config
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding()
        
        # Transformer layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dropout=config.dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.d_model, len(config.input_features))
        
    def create_positional_encoding(self) -> torch.Tensor:
        """Create positional encodings for the input sequence."""
        position = torch.arange(self.config.sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.config.d_model, 2) * 
                           (-torch.log(torch.tensor(10000.0)) / self.config.d_model))
        pos_encoding = torch.zeros(self.config.sequence_length, self.config.d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer."""
        # Add positional encoding
        x = x + self.positional_encoding.to(x.device)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Get the last sequence element
        x = x[-1]
        
        # Project to output dimension
        output = self.output_layer(x)
        
        return output
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer for PyTorch Lightning."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class TransformerModel(BaseModel):
    """Transformer model wrapper implementing BaseModel interface."""
    
    def __init__(self, config: TransformerConfig):
        """Initialize transformer model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for transformer model."""
        # Extract features and target
        features = data[self.config.input_features].values
        target = data[self.config.target_feature].values
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.config.sequence_length)
        
        return X, y
    
    def build_model(self) -> TimeSeriesTransformer:
        """Build transformer model architecture."""
        model = TimeSeriesTransformer(self.config)
        return model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the transformer model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "d_model": self.config.d_model,
            "num_heads": self.config.num_heads,
            "num_encoder_layers": self.config.num_encoder_layers,
            "dropout_rate": self.config.dropout_rate,
            "learning_rate": self.config.learning_rate
        })
        
        # Preprocess data
        X_train, y_train = self.preprocess_data(data)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        # Prepare validation data if provided
        val_loader = None
        if validation_data is not None:
            X_val, y_val = self.preprocess_data(validation_data)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size
            )
        
        # Initialize model and trainer
        self.model = self.build_model()
        trainer = pl.Trainer(
            max_epochs=self.config.epochs,
            accelerator='auto',
            devices=1,
            enable_progress_bar=True
        )
        
        # Train the model
        trainer.fit(
            self.model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader
        )
        
        # Log the model
        mlflow.pytorch.log_model(self.model, "model")
        
        # End MLflow run
        mlflow.end_run()
        
        return trainer.callback_metrics
    
    def predict(self, data: pd.DataFrame, horizon: int = 13) -> np.ndarray:
        """Generate predictions using the trained transformer model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        with torch.no_grad():
            # Preprocess input data
            features_scaled = self.scaler.transform(data[self.config.input_features].values)
            
            # Generate predictions
            predictions = []
            current_sequence = features_scaled[-self.config.sequence_length:]
            
            for _ in range(horizon):
                # Prepare input
                current_sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                
                # Get prediction
                next_pred = self.model(current_sequence_tensor).numpy()
                predictions.append(next_pred[0])
                
                # Update sequence
                current_sequence = np.vstack((current_sequence[1:], next_pred))
            
            # Inverse transform predictions
            predictions = np.array(predictions)
            predictions_rescaled = self.scaler.inverse_transform(predictions)
            
            return predictions_rescaled
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save transformer model to Azure ML workspace."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if model_name is None:
            model_name = f"{self.config.name}_model"
            
        # Save model locally first
        local_path = f"./tmp/{model_name}"
        torch.save(self.model.state_dict(), f"{local_path}/model.pt")
        
        # Register model in Azure ML workspace
        model = Model(
            path=local_path,
            name=model_name,
            description="Transformer model for sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load transformer model from Azure ML workspace."""
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
        
        # Initialize and load the model
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(f"./tmp/{model_name}/model.pt"))
        self.model.eval()
