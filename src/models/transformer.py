"""
Transformer model for sales forecasting.
"""
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import mlflow.pytorch
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

class TimeSeriesTransformer(pl.LightningModule):
    """Transformer model for time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        learning_rate: float
    ):
        """Initialize transformer model."""
        super().__init__()
        
        self.save_hyperparameters()
        
        # Embedding layers
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._create_positional_encoding()
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
    
    def _create_positional_encoding(self) -> torch.Tensor:
        """Create positional encoding matrix."""
        max_len = 1000
        d_model = self.hparams.d_model
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pos_encoding = torch.zeros((max_len, d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(1)].unsqueeze(0)
        
        # Transformer encoder
        x = self.transformer_encoder(x)
        
        # Output layer
        x = self.output_layer(x[:, -1, :])
        
        return x
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('train_loss', loss)
        return loss
    
    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate
        )

class TransformerModel(BaseModel):
    """Transformer model wrapper."""
    
    def __init__(
        self,
        sequence_length: int = 52,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        """Initialize transformer model."""
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Set random seed
        pl.seed_everything(42)
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Preprocess data for transformer model."""
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
                self.feature_scaler = torch.nn.BatchNorm1d(6)
                self.target_scaler = torch.nn.BatchNorm1d(1)
                
                scaled_features = self.feature_scaler(torch.FloatTensor(features))
                scaled_target = self.target_scaler(torch.FloatTensor(target).reshape(-1, 1))
            else:
                scaled_features = self.feature_scaler(torch.FloatTensor(features))
                scaled_target = self.target_scaler(torch.FloatTensor(target).reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_features) - self.sequence_length):
                X.append(scaled_features[i:i + self.sequence_length])
                y.append(scaled_target[i + self.sequence_length])
            
            return torch.stack(X), torch.stack(y)
        
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def create_model(self) -> TimeSeriesTransformer:
        """Create transformer model."""
        try:
            model = TimeSeriesTransformer(
                input_dim=6,
                d_model=self.d_model,
                nhead=self.nhead,
                num_layers=self.num_layers,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_rate,
                learning_rate=self.learning_rate
            )
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """Train transformer model."""
        try:
            X_train, y_train = train_data
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            val_loader = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_dataset = TensorDataset(X_val, y_val)
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size
                )
            
            # Configure trainer
            trainer = pl.Trainer(
                max_epochs=self.epochs,
                accelerator='auto',
                enable_progress_bar=True
            )
            
            # Train model
            trainer.fit(
                self.model,
                train_loader,
                val_loader
            )
            
            # Get metrics
            metrics = {
                'train_loss': trainer.callback_metrics['train_loss'].item()
            }
            
            if validation_data:
                metrics['val_loss'] = trainer.callback_metrics['val_loss'].item()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict_model(
        self,
        data: Tuple[torch.Tensor, torch.Tensor],
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            X, _ = data
            
            # Generate predictions iteratively
            predictions = []
            current_sequence = X[-1]
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(horizon):
                    # Predict next value
                    pred = self.model(current_sequence.unsqueeze(0))
                    predictions.append(pred.item())
                    
                    # Update sequence
                    current_sequence = torch.roll(current_sequence, -1, dims=0)
                    current_sequence[-1, 2:] = pred  # Update features with prediction
            
            # Inverse transform predictions
            predictions = torch.FloatTensor(predictions).reshape(-1, 1)
            predictions = self.target_scaler(predictions)
            
            return predictions.numpy().flatten()
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
