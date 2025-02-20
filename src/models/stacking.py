"""
Stacking meta-model for ensemble forecasting.
"""
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from azure.ai.ml.automl import (
    forecasting_settings,
    ForecastingSettings
)
import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)

class StackingMetaModel(BaseModel):
    """Stacking meta-model combining predictions from multiple base models."""
    
    def __init__(
        self,
        base_models: List[BaseModel],
        meta_model_type: str = 'xgboost',
        learning_rate: float = 0.1,
        max_depth: int = 5,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        **kwargs
    ):
        """Initialize stacking meta-model."""
        super().__init__(**kwargs)
        self.base_models = base_models
        self.meta_model_type = meta_model_type
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
    
    def _get_base_predictions(
        self,
        data: pd.DataFrame,
        horizon: Optional[int] = None
    ) -> np.ndarray:
        """Get predictions from all base models."""
        try:
            predictions = []
            
            for model in self.base_models:
                if horizon is None:
                    # Training phase - use actual data
                    processed_data = model.preprocess_data(data)
                    model_preds = model.predict(processed_data)
                else:
                    # Prediction phase - generate future predictions
                    processed_data = model.preprocess_data(data)
                    model_preds = model.predict(processed_data, horizon)
                
                predictions.append(model_preds)
            
            return np.column_stack(predictions)
        
        except Exception as e:
            logger.error(f"Failed to get base predictions: {str(e)}")
            raise
    
    def preprocess_data(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess data for meta-model."""
        try:
            # Get base model predictions
            base_predictions = self._get_base_predictions(data)
            
            if is_training:
                # For training, we need the actual target values
                target = data['CV Gross Profit'].values
                
                # Scale target
                self.target_scaler = StandardScaler()
                scaled_target = self.target_scaler.fit_transform(target.reshape(-1, 1))
                
                return base_predictions, scaled_target.flatten()
            else:
                return base_predictions, None
        
        except Exception as e:
            logger.error(f"Failed to preprocess data: {str(e)}")
            raise
    
    def create_model(self) -> Any:
        """Create meta-model."""
        try:
            if self.meta_model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    n_estimators=self.n_estimators,
                    subsample=self.subsample,
                    colsample_bytree=self.colsample_bytree,
                    objective='reg:squarederror',
                    random_state=42
                )
            else:
                raise ValueError(f"Unsupported meta-model type: {self.meta_model_type}")
            
            return model
        
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            raise
    
    def train_model(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, float]:
        """Train meta-model."""
        try:
            X_train, y_train = train_data
            
            if validation_data:
                X_val, y_val = validation_data
            else:
                # Split training data if no validation set provided
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train, y_train,
                    test_size=0.2,
                    random_state=42
                )
            
            # Train model
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Calculate metrics
            train_pred = self.model.predict(X_train)
            val_pred = self.model.predict(X_val)
            
            metrics = {
                'train_mse': float(np.mean((y_train - train_pred) ** 2)),
                'train_mae': float(np.mean(np.abs(y_train - train_pred))),
                'val_mse': float(np.mean((y_val - val_pred) ** 2)),
                'val_mae': float(np.mean(np.abs(y_val - val_pred)))
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Failed to train model: {str(e)}")
            raise
    
    def predict_model(
        self,
        data: Tuple[np.ndarray, Optional[np.ndarray]],
        horizon: int = 52
    ) -> np.ndarray:
        """Generate predictions."""
        try:
            # Get base model predictions for future horizon
            base_predictions = self._get_base_predictions(data[0], horizon)
            
            # Generate meta-model predictions
            predictions = self.model.predict(base_predictions)
            
            # Inverse transform predictions
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            )
            
            return predictions.flatten()
        
        except Exception as e:
            logger.error(f"Failed to generate predictions: {str(e)}")
            raise
    
    def train_with_automl(
        self,
        ml_client: MLClient,
        data: pd.DataFrame,
        compute_name: str,
        experiment_name: str = "stacking_automl",
        **kwargs
    ) -> Dict[str, float]:
        """Train stacking model using Azure AutoML."""
        try:
            # Get base model predictions
            base_predictions = self._get_base_predictions(data)
            
            # Create training dataset
            train_data = pd.DataFrame(
                base_predictions,
                columns=[f'model_{i}_pred' for i in range(len(self.base_models))]
            )
            train_data['ds'] = data.apply(
                lambda x: datetime(
                    int(x['Transaction Year']),
                    1,
                    1
                ) + pd.Timedelta(weeks=int(x['Transaction Week'])-1),
                axis=1
            )
            train_data['y'] = data['CV Gross Profit']
            
            # Configure AutoML settings
            training_settings = ForecastingSettings(
                time_column_name="ds",
                target_column_name="y",
                time_series_id_column_names=None,  # Single time series
                forecast_horizon=52,
                feature_column_names=[f'model_{i}_pred' for i in range(len(self.base_models))],
                allowed_training_algorithms=["XGBoostRegressor"],
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
                display_name="stacking_training"
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
    
    def get_model_importances(self) -> pd.DataFrame:
        """Get importance scores for base models."""
        try:
            if self.meta_model_type != 'xgboost':
                raise ValueError("Feature importance only available for XGBoost meta-model")
            
            # Get feature importance scores
            importances = self.model.feature_importances_
            
            # Create DataFrame
            importance_df = pd.DataFrame({
                'model': [model.__class__.__name__ for model in self.base_models],
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
        
        except Exception as e:
            logger.error(f"Failed to get model importances: {str(e)}")
            raise
