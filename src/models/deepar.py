"""
DeepAR+ model for probabilistic sales forecasting.
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
import torch
import mlflow
import logging
from datetime import datetime

from .base import BaseModel

logger = logging.getLogger(__name__)

class DeepARPlusModel(BaseModel):
    """DeepAR+ model for probabilistic forecasting."""
    
    def __init__(
        self,
        prediction_length: int = 52,
        context_length: int = 52,
        num_layers: int = 2,
        hidden_size: int = 40,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        epochs: int = 100,
        num_batches_per_epoch: int = 50,
        **kwargs
    ):
        """Initialize DeepAR+ model."""
        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        
        # Set random seed
        torch.manual_seed(42)
    
    def _create_gluonts_dataset(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> ListDataset:
        """Create GluonTS dataset."""
        try:
            # Group data by time series ID if multiple series exist
            if 'series_id' in data.columns:
                grouped_data = data.groupby('series_id')
            else:
                # If no series_id, treat as single time series
                grouped_data = [(None, data)]
            
            # Create list of dictionaries for GluonTS
            series_list = []
            for series_id, group in grouped_data:
                # Sort by time
                group = group.sort_values(['Transaction Year', 'Transaction Week'])
                
                # Create timestamp
                group['timestamp'] = group.apply(
                    lambda x: datetime(
                        int(x['Transaction Year']),
                        1,
                        1
                    ) + pd.Timedelta(weeks=int(x['Transaction Week'])-1),
                    axis=1
                )
                
                # Create dictionary for this series
                series_dict = {
                    'target': group['CV Gross Profit'].values,
                    'start': group['timestamp'].iloc[0],
                    'feat_dynamic_real': [
                        group['Volume'].values,
                        group['CV Gross Sales'].values,
                        group['CV Net Sales'].values,
                        group['CV COGS'].values
                    ]
                }
                
                if series_id is not None:
                    series_dict['item_id'] = str(series_id)
                
                series_list.append(series_dict)
            
            # Create GluonTS dataset
            return ListDataset(
                series_list,
                freq='W'  # Weekly frequency
            )
        
        except Exception as e:
            logger.error(f"Failed to create GluonTS dataset: {str(e)}")
            raise
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> ListDataset:
        """Preprocess data for DeepAR+ model."""
        try:
            # Create GluonTS dataset
            dataset = self._create_gluonts_dataset(data, is_training)
            
            return dataset
        
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def create_model(self) -> DeepAREstimator:
        """Create DeepAR+ model."""
        try:
            # Configure trainer
            trainer = Trainer(
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                num_batches_per_epoch=self.num_batches_per_epoch,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Create model
            model = DeepAREstimator(
                freq='W',  # Weekly frequency
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                dropout_rate=self.dropout_rate,
                trainer=trainer,
                use_feat_dynamic_real=True
            )
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_data: ListDataset,
        validation_data: Optional[ListDataset] = None
    ) -> Dict[str, float]:
        """Train DeepAR+ model."""
        try:
            # Train model
            predictor = self.model.train(train_data)
            
            # Store predictor
            self.predictor = predictor
            
            # Calculate metrics
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=validation_data if validation_data else train_data,
                predictor=predictor,
                num_samples=100
            )
            
            # Convert iterators to lists
            forecasts = list(forecast_it)
            tss = list(ts_it)
            
            # Calculate metrics
            metrics = {}
            
            # Mean metrics across all series
            mean_forecasts = np.array([f.mean for f in forecasts])
            actual_values = np.array([ts.values[-self.prediction_length:] for ts in tss])
            
            # Calculate MSE
            mse = np.mean((mean_forecasts - actual_values) ** 2)
            metrics['mse'] = float(mse)
            
            # Calculate MAE
            mae = np.mean(np.abs(mean_forecasts - actual_values))
            metrics['mae'] = float(mae)
            
            # Calculate RMSE
            rmse = np.sqrt(mse)
            metrics['rmse'] = float(rmse)
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict_model(
        self,
        data: ListDataset,
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            if not hasattr(self, 'predictor'):
                raise ValueError("Model must be trained before prediction")
            
            # Generate forecasts
            forecast_it, _ = make_evaluation_predictions(
                dataset=data,
                predictor=self.predictor,
                num_samples=100
            )
            
            # Get mean predictions
            forecasts = list(forecast_it)
            predictions = np.array([f.mean for f in forecasts])
            
            # If multiple series, return average prediction
            if predictions.ndim > 1:
                predictions = predictions.mean(axis=0)
            
            # Trim to requested horizon
            predictions = predictions[:horizon]
            
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
    
    def get_prediction_intervals(
        self,
        data: ListDataset,
        horizon: int = 52,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Get prediction intervals."""
        try:
            if not hasattr(self, 'predictor'):
                raise ValueError("Model must be trained before prediction")
            
            # Generate forecasts
            forecast_it, _ = make_evaluation_predictions(
                dataset=data,
                predictor=self.predictor,
                num_samples=100
            )
            
            forecasts = list(forecast_it)
            
            # Calculate intervals
            lower_percentile = (1 - confidence_level) / 2
            upper_percentile = 1 - lower_percentile
            
            intervals = {
                'mean': np.array([f.mean for f in forecasts]),
                'lower': np.array([f.quantile(lower_percentile) for f in forecasts]),
                'upper': np.array([f.quantile(upper_percentile) for f in forecasts])
            }
            
            # If multiple series, return average intervals
            if intervals['mean'].ndim > 1:
                intervals = {
                    k: v.mean(axis=0)[:horizon]
                    for k, v in intervals.items()
                }
            else:
                intervals = {
                    k: v[:horizon]
                    for k, v in intervals.items()
                }
            
            return intervals
        
        except Exception as e:
            logger.error(f"Failed to generate prediction intervals: {str(e)}")
            raise
