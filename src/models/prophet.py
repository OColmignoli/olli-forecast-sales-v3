"""
Prophet model for sales forecasting with Azure AutoML integration.
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from prophet import Prophet
from azure.ai.ml import MLClient
from azure.ai.ml.automl import (
    forecasting_settings,
    ForecastingSettings
)
from azure.ai.ml import Input, Output
import mlflow.prophet
import logging
from datetime import datetime

from .base import BaseModel

logger = logging.getLogger(__name__)

class ProphetModel(BaseModel):
    """Prophet model with Azure AutoML integration."""
    
    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        holidays_prior_scale: float = 10.0,
        changepoint_range: float = 0.8,
        **kwargs
    ):
        """Initialize Prophet model."""
        super().__init__(**kwargs)
        self.seasonality_mode = seasonality_mode
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.changepoint_range = changepoint_range
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> pd.DataFrame:
        """Preprocess data for Prophet model."""
        try:
            # Create datetime index
            data['ds'] = data.apply(
                lambda x: datetime(
                    int(x['Transaction Year']),
                    1,
                    1
                ) + pd.Timedelta(weeks=int(x['Transaction Week'])-1),
                axis=1
            )
            
            # Rename target column
            data['y'] = data['CV Gross Profit']
            
            # Add additional regressors
            data['volume'] = data['Volume']
            data['gross_sales'] = data['CV Gross Sales']
            data['net_sales'] = data['CV Net Sales']
            data['cogs'] = data['CV COGS']
            
            return data[['ds', 'y', 'volume', 'gross_sales', 'net_sales', 'cogs']]
        
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def create_model(self) -> Prophet:
        """Create Prophet model."""
        try:
            model = Prophet(
                seasonality_mode=self.seasonality_mode,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                daily_seasonality=self.daily_seasonality,
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                holidays_prior_scale=self.holidays_prior_scale,
                changepoint_range=self.changepoint_range
            )
            
            # Add regressors
            model.add_regressor('volume')
            model.add_regressor('gross_sales')
            model.add_regressor('net_sales')
            model.add_regressor('cogs')
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """Train Prophet model."""
        try:
            # Fit model
            self.model.fit(train_data)
            
            # Calculate metrics
            metrics = {}
            
            if validation_data is not None:
                # Make predictions on validation set
                forecast = self.model.predict(validation_data)
                
                # Calculate metrics
                y_true = validation_data['y'].values
                y_pred = forecast['yhat'].values
                
                mse = np.mean((y_true - y_pred) ** 2)
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(mse)
                
                metrics.update({
                    'val_mse': float(mse),
                    'val_mae': float(mae),
                    'val_rmse': float(rmse)
                })
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict_model(
        self,
        data: pd.DataFrame,
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=horizon,
                freq='W'
            )
            
            # Add regressor values
            last_values = data.iloc[-1]
            for regressor in ['volume', 'gross_sales', 'net_sales', 'cogs']:
                future[regressor] = last_values[regressor]
            
            # Make predictions
            forecast = self.model.predict(future)
            
            # Get predictions for horizon
            predictions = forecast['yhat'].values[-horizon:]
            
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
    
    def train_with_automl(
        self,
        ml_client: MLClient,
        data: pd.DataFrame,
        compute_name: str,
        experiment_name: str = "prophet_automl",
        **kwargs
    ) -> Dict[str, float]:
        """Train Prophet model using Azure AutoML."""
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)
            
            # Configure AutoML settings
            training_settings = ForecastingSettings(
                time_column_name="ds",
                target_column_name="y",
                time_series_id_column_names=None,  # Single time series
                forecast_horizon=52,
                feature_column_names=[
                    'volume',
                    'gross_sales',
                    'net_sales',
                    'cogs'
                ],
                allowed_training_algorithms=["Prophet"],
                primary_metric="normalized_root_mean_squared_error",
                enable_dnn_training=False,
                validation_size=0.2,
                n_cross_validations=5,
                **kwargs
            )
            
            # Create AutoML job
            automl_job = ml_client.automl.create_or_update(
                automl_settings=training_settings,
                compute_name=compute_name,
                experiment_name=experiment_name,
                display_name="prophet_training"
            )
            
            # Wait for job completion
            result = automl_job.wait_for_completion()
            
            # Get best model
            best_model = result.get_best_model()
            
            # Update current model with best model
            self.model = best_model
            self.is_trained = True
            
            # Get metrics
            metrics = result.get_metrics()
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train with AutoML: {str(e)}")
            raise
    
    def get_component_analysis(self) -> Dict[str, pd.DataFrame]:
        """Get Prophet component analysis."""
        try:
            # Get forecast with components
            forecast = self.model.predict(self.model.history)
            
            components = {}
            
            # Trend
            components['trend'] = pd.DataFrame({
                'ds': forecast['ds'],
                'trend': forecast['trend']
            })
            
            # Weekly seasonality
            if self.weekly_seasonality:
                components['weekly'] = pd.DataFrame({
                    'ds': forecast['ds'],
                    'weekly': forecast['weekly']
                })
            
            # Yearly seasonality
            if self.yearly_seasonality:
                components['yearly'] = pd.DataFrame({
                    'ds': forecast['ds'],
                    'yearly': forecast['yearly']
                })
            
            # Regressors
            for regressor in ['volume', 'gross_sales', 'net_sales', 'cogs']:
                components[f'regressor_{regressor}'] = pd.DataFrame({
                    'ds': forecast['ds'],
                    regressor: forecast[f'extra_regressors_{regressor}']
                })
            
            return components
        
        except Exception as e:
            logger.error(f"Failed to get component analysis: {str(e)}")
            raise
