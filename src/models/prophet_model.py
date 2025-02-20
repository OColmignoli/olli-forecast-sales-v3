"""Prophet model implementation for time series forecasting."""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
import mlflow
import mlflow.prophet
import joblib

from .base_model import BaseModel
from .config import ProphetConfig

class ProphetModel(BaseModel):
    """Prophet implementation using Facebook Prophet."""
    
    def __init__(self, config: ProphetConfig):
        """Initialize Prophet model with configuration."""
        super().__init__(config)
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data for Prophet model."""
        # Prophet requires 'ds' (date) and 'y' (target) columns
        df = data.copy()
        
        # Create datetime index if not present
        if 'ds' not in df.columns:
            # Assuming data is weekly and starts from the first week
            df['ds'] = pd.date_range(
                start='2020-01-01',  # This will be adjusted based on actual data
                periods=len(df),
                freq='W-MON'
            )
        
        # Scale the target variable
        if 'y' not in df.columns:
            df['y'] = df[self.config.target_feature]
            self.scaler.fit(df[['y']])
            df['y'] = self.scaler.transform(df[['y']])
        
        return df[['ds', 'y']]
    
    def build_model(self) -> Prophet:
        """Build Prophet model with configuration."""
        model = Prophet(
            seasonality_mode=self.config.seasonality_mode,
            yearly_seasonality=self.config.yearly_seasonality,
            weekly_seasonality=self.config.weekly_seasonality,
            daily_seasonality=self.config.daily_seasonality,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale
        )
        
        return model
    
    def train(self, data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Train the Prophet model."""
        # Start MLflow run
        mlflow.start_run()
        
        # Log parameters
        mlflow.log_params({
            "seasonality_mode": self.config.seasonality_mode,
            "yearly_seasonality": self.config.yearly_seasonality,
            "weekly_seasonality": self.config.weekly_seasonality,
            "changepoint_prior_scale": self.config.changepoint_prior_scale,
            "seasonality_prior_scale": self.config.seasonality_prior_scale
        })
        
        # Preprocess data
        df = self.preprocess_data(data)
        
        # Build and train model
        self.model = self.build_model()
        self.model.fit(df)
        
        # Make in-sample predictions for evaluation
        forecast = self.model.predict(df)
        
        # Calculate metrics
        metrics = self.evaluate(
            df['y'].values,
            forecast['yhat'].values
        )
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Log model
        mlflow.prophet.log_model(self.model, "model")
        
        # End MLflow run
        mlflow.end_run()
        
        return metrics
    
    def predict(self, data: pd.DataFrame, horizon: int = 13) -> np.ndarray:
        """Generate predictions using the trained Prophet model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Create future dataframe
        last_date = data['ds'].max() if 'ds' in data.columns else pd.Timestamp('2020-01-01')
        future = pd.DataFrame({
            'ds': pd.date_range(
                start=last_date,
                periods=horizon + 1,
                freq='W-MON'
            )[1:]  # Exclude the last known date
        })
        
        # Make predictions
        forecast = self.model.predict(future)
        predictions = forecast['yhat'].values.reshape(-1, 1)
        
        # Get prediction intervals
        predictions_rescaled = self.scaler.inverse_transform(predictions)
        
        return predictions_rescaled
    
    def get_forecast_components(self, forecast: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract forecast components (trend, seasonality, etc.)."""
        components = {}
        
        if 'trend' in forecast.columns:
            components['trend'] = self.scaler.inverse_transform(
                forecast['trend'].values.reshape(-1, 1)
            )
        
        if 'yearly' in forecast.columns:
            components['yearly_seasonality'] = forecast['yearly'].values
            
        if 'weekly' in forecast.columns:
            components['weekly_seasonality'] = forecast['weekly'].values
            
        if 'multiplicative_terms' in forecast.columns:
            components['multiplicative_terms'] = forecast['multiplicative_terms'].values
            
        return components
    
    def save_model(self, model_name: Optional[str] = None) -> str:
        """Save Prophet model to Azure ML workspace."""
        if self.model is None:
            raise ValueError("No model to save")
            
        if model_name is None:
            model_name = f"{self.config.name}_model"
            
        # Save model locally first
        local_path = f"./tmp/{model_name}"
        
        # Save both the model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, f"{local_path}/model.joblib")
        
        # Register model in Azure ML workspace
        model = Model(
            path=local_path,
            name=model_name,
            description="Prophet model for sales forecasting",
            type=AssetTypes.CUSTOM_MODEL
        )
        
        self.ml_client.models.create_or_update(model)
        
        return model_name
    
    def load_model(self, model_name: Optional[str] = None) -> None:
        """Load Prophet model from Azure ML workspace."""
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
        
        # Load the model and scaler
        model_data = joblib.load(f"./tmp/{model_name}/model.joblib")
        self.model = model_data['model']
        self.scaler = model_data['scaler']
