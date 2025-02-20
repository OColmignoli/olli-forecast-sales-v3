"""Azure ML pipeline for automated model training."""
from typing import Dict, Any, Optional, List
import os
from azure.ai.ml import MLClient, Input, Output, dsl
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential

def get_ml_client() -> MLClient:
    """Get Azure ML client."""
    return MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=os.getenv('AZURE_SUBSCRIPTION_ID'),
        resource_group_name=os.getenv('AZURE_RESOURCE_GROUP'),
        workspace_name=os.getenv('AZURE_ML_WORKSPACE')
    )

@dsl.pipeline(
    name="sales_forecast_training",
    description="Pipeline for training sales forecasting models",
    default_compute="cpu-cluster"
)
def create_training_pipeline(
    data_path: Input(type=AssetTypes.URI_FILE),
    model_output: Output(type=AssetTypes.MLFLOW_MODEL)
) -> None:
    """Create training pipeline."""
    
    # Data validation component
    @dsl.component(
        name="data_validation",
        display_name="Validate Input Data",
        description="Validates the input data format and quality"
    )
    def validate_data(
        data: Input(type=AssetTypes.URI_FILE),
        validated_data: Output(type=AssetTypes.URI_FILE)
    ) -> None:
        import pandas as pd
        from utils.data_preprocessing import validate_columns, validate_data_types, validate_value_ranges
        
        # Read data
        df = pd.read_csv(data)
        
        # Validate data
        if not validate_columns(df):
            raise ValueError("Invalid columns in input data")
        
        if not validate_data_types(df):
            raise ValueError("Invalid data types in input data")
        
        if not validate_value_ranges(df):
            raise ValueError("Invalid value ranges in input data")
        
        # Save validated data
        df.to_csv(validated_data, index=False)
    
    # Feature engineering component
    @dsl.component(
        name="feature_engineering",
        display_name="Feature Engineering",
        description="Creates features for model training"
    )
    def create_features(
        data: Input(type=AssetTypes.URI_FILE),
        features: Output(type=AssetTypes.URI_FILE)
    ) -> None:
        import pandas as pd
        from utils.data_preprocessing import (
            create_date_features,
            create_lag_features,
            create_rolling_features,
            handle_missing_values
        )
        
        # Read data
        df = pd.read_csv(data)
        
        # Create features
        df = create_date_features(df)
        df = create_lag_features(df, ['Volume', 'CV Net Sales'], [1, 2, 3, 4, 8, 13, 26, 52])
        df = create_rolling_features(df, ['Volume', 'CV Net Sales'], [4, 8, 13, 26, 52])
        df = handle_missing_values(df)
        
        # Save features
        df.to_csv(features, index=False)
    
    # Model training component
    @dsl.component(
        name="model_training",
        display_name="Train Models",
        description="Trains all models and creates stacking ensemble"
    )
    def train_models(
        features: Input(type=AssetTypes.URI_FILE),
        model: Output(type=AssetTypes.MLFLOW_MODEL)
    ) -> None:
        import pandas as pd
        import mlflow
        from models.stacking_model import StackingModel
        from models.config import StackingConfig
        
        # Read features
        df = pd.read_csv(features)
        
        # Configure and train stacking model
        config = StackingConfig(
            base_models=['lstm', 'transformer', 'deepar', 'cnn', 'prophet'],
            meta_model='xgboost'
        )
        
        model = StackingModel(config)
        model.train(df)
        
        # Log model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=model
        )
    
    # Model evaluation component
    @dsl.component(
        name="model_evaluation",
        display_name="Evaluate Model",
        description="Evaluates model performance"
    )
    def evaluate_model(
        features: Input(type=AssetTypes.URI_FILE),
        model: Input(type=AssetTypes.MLFLOW_MODEL),
        metrics: Output(type=AssetTypes.URI_FILE)
    ) -> None:
        import pandas as pd
        import json
        import mlflow
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Load data and model
        df = pd.read_csv(features)
        model = mlflow.pyfunc.load_model(model)
        
        # Generate predictions
        predictions = model.predict(df)
        
        # Calculate metrics
        metrics_dict = {
            'mse': mean_squared_error(df['CV Net Sales'], predictions),
            'mae': mean_absolute_error(df['CV Net Sales'], predictions),
            'r2': r2_score(df['CV Net Sales'], predictions)
        }
        
        # Save metrics
        with open(metrics, 'w') as f:
            json.dump(metrics_dict, f)
    
    # Pipeline steps
    validated_data = validate_data(data_path)
    features = create_features(validated_data.outputs.validated_data)
    trained_model = train_models(features.outputs.features)
    evaluate_model(
        features.outputs.features,
        trained_model.outputs.model,
        model_output
    )

def create_and_submit_pipeline() -> None:
    """Create and submit training pipeline."""
    # Get ML client
    ml_client = get_ml_client()
    
    # Create compute cluster if not exists
    from azure.ai.ml.entities import AmlCompute
    
    cpu_compute_target = "cpu-cluster"
    try:
        compute_cluster = ml_client.compute.get(cpu_compute_target)
    except Exception:
        compute_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size="Standard_DS3_v2",
            min_instances=0,
            max_instances=4,
            idle_time_before_scale_down=120
        )
        ml_client.compute.begin_create_or_update(compute_cluster)
    
    # Create environment
    env = Environment(
        name="sales-forecast-env",
        description="Environment for sales forecasting",
        build=BuildContext(path=".")
    )
    ml_client.environments.create_or_update(env)
    
    # Create pipeline
    pipeline = create_training_pipeline(
        data_path=Input(type=AssetTypes.URI_FILE),
        model_output=Output(type=AssetTypes.MLFLOW_MODEL)
    )
    
    # Submit pipeline
    pipeline_job = ml_client.jobs.create_or_update(
        pipeline,
        experiment_name="sales_forecast_training"
    )
    
    return pipeline_job

if __name__ == "__main__":
    pipeline_job = create_and_submit_pipeline()
    print(f"Pipeline submitted. Job ID: {pipeline_job.id}")
