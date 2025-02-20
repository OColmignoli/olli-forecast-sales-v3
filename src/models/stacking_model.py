"""Stacking model implementation for ensemble forecasting."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import joblib

from .base_model import BaseModel
from .config import StackingConfig
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .deepar_model import DeepARModel
from .cnn_model import CNNModel
from .prophet_model import ProphetModel

class StackingModel(BaseModel):
    """Stacking ensemble implementation."""
    
    def __init__(self, config: StackingConfig):
        """Initialize stacking model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        self.base_models = {}
        self.meta_model = None
        
    def initialize_base_models(self) -> None:
        """Initialize all base models."""
        model_classes = {
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'deepar': DeepARModel,
            'cnn': CNNModel,
            'prophet': ProphetModel
        }
        
        for model_name in self.config.base_models:
            if model_name not in model_classes:
                raise ValueError(f"Unknown model type: {model_name}")
            
            # Initialize model with its specific config
            model_class = model_classes[model_name]
            self.base_models[model_name] = model_class(self.config)
    
    def initialize_meta_model(self) -> None:
        """Initialize meta-model (XGBoost or Random Forest)."""
        if self.config.meta_model == 'xgboost':
            self.meta_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                objective='reg:squarederror'
            )
        elif self.config.meta_model == 'randomforest':
            self.meta_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta-model type: {self.config.meta_model}")
    
    def generate_base_predictions(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions from all base models."""
        predictions = []
        
        for model_name, model in self.base_models.items():
            try:
                model_preds = model.predict(data)
                if isinstance(model_preds, dict):  # Handle probabilistic predictions
                    model_preds = model_preds['mean']
                predictions.append(model_preds)
            except Exception as e:
                print(f"Error getting predictions from {model_name}: {str(e)}")
                # Use zeros as fallback
                predictions.append(np.zeros((self.config.forecast_horizon, len(self.config.input_features))))
        
        return np.hstack(predictions)
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess data for stacking model."""
        # Scale the target variable
        target = data[self.config.target_feature].values
        target_scaled = self.scaler.fit_transform(target.reshape(-1, 1))
        
        # Generate base model predictions
        base_predictions = self.generate_base_predictions(data)
        
        return base_predictions, target_scaled
    
    def build_model(self) -> Any:
        """Build stacking model architecture."""
        self.initialize_base_models()
        self.initialize_meta_model()
        return self.meta_model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the stacking model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "base_models": self.config.base_models,
            "meta_model": self.config.meta_model,
            "cv_folds": self.config.cv_folds
        })
        
        # Train base models first
        print("Training base models...")
        for model_name, model in self.base_models.items():
            print(f"Training {model_name}...")
            model.train(data, validation_data)
        
        # Generate predictions from base models
        print("Generating base model predictions...")
        X, y = self.preprocess_data(data)
        
        # Initialize and train meta-model
        print("Training meta-model...")
        self.model = self.build_model()
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            self.model.fit(X_train, y_train.ravel())
            score = self.model.score(X_val, y_val.ravel())
            cv_scores.append(score)
        
        # Final fit on all data
        self.model.fit(X, y.ravel())
        
        # Calculate and log metrics
        metrics = {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores)
        }
        
        mlflow.log_metrics(metrics)
        
        # Log meta-model
        if isinstance(self.model, XGBRegressor):
            mlflow.xgboost.log_model(self.model, "meta_model")
        else:
            mlflow.sklearn.log_model(self.model, "meta_model")
        
        # End MLflow run
        mlflow.end_run()
        
        return metrics
    
    def predict(self, data: pd.DataFrame, horizon: int = 13) -> np.ndarray:
        """Generate predictions using the stacking model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Generate base model predictions
        base_predictions = self.generate_base_predictions(data)
        
        # Make predictions with meta-model
        predictions = self.model.predict(base_predictions)
        
        # Reshape and inverse transform predictions
        predictions = predictions.reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        
        return predictions_rescaled
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save stacking model to Azure ML workspace."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if model_name is None:
            model_name = f"{self.config.name}_model"
            
        # Save model locally first
        local_path = f"./tmp/{model_name}"
        
        # Save meta-model and scaler
        model_data = {
            'meta_model': self.model,
            'scaler': self.scaler
        }
        
        # Save base models
        for model_name, model in self.base_models.items():
            model_data[f'base_model_{model_name}'] = model
        
        joblib.dump(model_data, f"{local_path}/model.joblib")
        
        # Register model in Azure ML workspace
        model = Model(
            path=local_path,
            name=model_name,
            description="Stacking ensemble model for sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load stacking model from Azure ML workspace."""
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
        
        # Load all models and scaler
        model_data = joblib.load(f"./tmp/{model_name}/model.joblib")
        
        self.model = model_data['meta_model']
        self.scaler = model_data['scaler']
        
        # Load base models
        self.base_models = {}
        for key, model in model_data.items():
            if key.startswith('base_model_'):
                model_name = key.replace('base_model_', '')
                self.base_models[model_name] = model
