"""DeepAR+ model implementation for probabilistic time series forecasting."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import mlflow.pytorch

from .base_model import BaseModel
from .config import DeepARConfig

class DeepARNet(pl.LightningModule):
    """PyTorch Lightning implementation of DeepAR+ network."""
    
    def __init__(self, config: DeepARConfig):
        """Initialize the DeepAR+ network."""
        super().__init__()
        self.config = config
        
        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=len(config.input_features),
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=0.1,
            batch_first=True
        )
        
        # Output layers for probabilistic forecasting
        self.mu_layer = nn.Linear(config.hidden_size, 1)
        self.sigma_layer = nn.Linear(config.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network."""
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        
        # Get last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        # Generate distribution parameters
        mu = self.mu_layer(last_hidden)
        sigma = F.softplus(self.sigma_layer(last_hidden)) + 1e-6
        
        return mu, sigma
    
    def gaussian_likelihood(self, mu: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the negative log likelihood of target under Gaussian distribution."""
        dist = torch.distributions.Normal(mu, sigma)
        return -dist.log_prob(target).mean()
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step for PyTorch Lightning."""
        x, y = batch
        mu, sigma = self(x)
        loss = self.gaussian_likelihood(mu, sigma, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step for PyTorch Lightning."""
        x, y = batch
        mu, sigma = self(x)
        loss = self.gaussian_likelihood(mu, sigma, y)
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

class DeepARModel(BaseModel):
    """DeepAR+ model wrapper implementing BaseModel interface."""
    
    def __init__(self, config: DeepARConfig):
        """Initialize DeepAR+ model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for DeepAR+ model."""
        # Extract features and target
        features = data[self.config.input_features].values
        target = data[self.config.target_feature].values
        
        # Scale the data
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, self.config.sequence_length)
        
        return X, y
    
    def build_model(self) -> DeepARNet:
        """Build DeepAR+ model architecture."""
        model = DeepARNet(self.config)
        return model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the DeepAR+ model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "num_layers": self.config.num_layers,
            "hidden_size": self.config.hidden_size,
            "context_length": self.config.context_length,
            "prediction_length": self.config.prediction_length,
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
    
    def predict(self, data: pd.DataFrame, horizon: int = 13, num_samples: int = 100) -> Dict[str, np.ndarray]:
        """Generate probabilistic predictions using the trained DeepAR+ model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        self.model.eval()
        with torch.no_grad():
            # Preprocess input data
            features_scaled = self.scaler.transform(data[self.config.input_features].values)
            
            # Generate predictions
            predictions = []
            prediction_intervals = []
            current_sequence = features_scaled[-self.config.sequence_length:]
            
            for _ in range(horizon):
                # Prepare input
                current_sequence_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
                
                # Get distribution parameters
                mu, sigma = self.model(current_sequence_tensor)
                
                # Generate samples from the predicted distribution
                dist = torch.distributions.Normal(mu, sigma)
                samples = dist.sample((num_samples,)).numpy()
                
                # Calculate mean and prediction intervals
                mean_pred = np.mean(samples, axis=0)
                lower_bound = np.percentile(samples, 5, axis=0)
                upper_bound = np.percentile(samples, 95, axis=0)
                
                predictions.append(mean_pred)
                prediction_intervals.append((lower_bound, upper_bound))
                
                # Update sequence
                current_sequence = np.vstack((current_sequence[1:], mean_pred))
            
            # Inverse transform predictions and intervals
            predictions = np.array(predictions).reshape(-1, len(self.config.input_features))
            predictions_rescaled = self.scaler.inverse_transform(predictions)
            
            lower_bounds = np.array([lb for lb, _ in prediction_intervals])
            upper_bounds = np.array([ub for _, ub in prediction_intervals])
            
            lower_bounds_rescaled = self.scaler.inverse_transform(lower_bounds.reshape(-1, len(self.config.input_features)))
            upper_bounds_rescaled = self.scaler.inverse_transform(upper_bounds.reshape(-1, len(self.config.input_features)))
            
            return {
                'mean': predictions_rescaled,
                'lower_bound': lower_bounds_rescaled,
                'upper_bound': upper_bounds_rescaled
            }
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save DeepAR+ model to Azure ML workspace."""
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
            description="DeepAR+ model for probabilistic sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load DeepAR+ model from Azure ML workspace."""
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
