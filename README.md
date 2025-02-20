# OLLI Sales Forecasting V3

Advanced sales forecasting system using Azure ML and multiple deep learning models.

## Features

- Multiple forecasting models:
  - LSTM (Long Short-Term Memory)
  - Transformer (TFT)
  - DeepAR+
  - CNN (Convolutional Neural Network)
  - Prophet
  - Stacking Meta-Model

- Azure ML Integration:
  - Model training and deployment
  - MLflow tracking
  - Azure AutoML support
  - Compute cluster management

- Web Interface:
  - Modern responsive design
  - Data upload/download
  - Model training controls
  - Interactive visualizations

## Project Structure

```
olli-forecast-sales-v3/
├── config/
│   ├── azure_config.yaml
│   └── conda/
│       ├── tensorflow.yml
│       ├── pytorch.yml
│       ├── gluonts.yml
│       └── prophet.yml
├── src/
│   ├── azure/
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── validation.py
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── lstm.py
│   │   ├── transformer.py
│   │   ├── deepar.py
│   │   ├── cnn.py
│   │   ├── prophet.py
│   │   └── stacking.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── data_pipeline.py
│   │   └── training_pipeline.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── model_registry.py
│   └── web/
│       ├── __init__.py
│       ├── api.py
│       └── frontend/
│           ├── src/
│           │   ├── App.js
│           │   └── components/
│           │       ├── DataUpload.js
│           │       ├── ModelTraining.js
│           │       ├── ForecastGeneration.js
│           │       └── ForecastVisualization.js
│           └── public/
└── tests/
    └── __init__.py
```

## Setup

1. Azure Configuration:
   - Create Azure ML workspace
   - Set up compute cluster
   - Configure storage account
   - Deploy web app

2. Environment Setup:
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Set up Azure credentials
   export AZURE_SUBSCRIPTION_ID="c828c783-7a28-48f4-b56f-a6c189437d77"
   export AZURE_RESOURCE_GROUP="OLLI-resource"
   export AZURE_WORKSPACE_NAME="OLLI_ML_Forecast"
   ```

3. Start Web Interface:
   ```bash
   # Start backend
   cd src/web
   uvicorn api:app --reload

   # Start frontend
   cd frontend
   npm install
   npm start
   ```

## Usage

1. Data Upload:
   - Upload sales data CSV file
   - Required columns:
     - Transaction Year (int)
     - Transaction Week (int, 1-53)
     - Volume (float)
     - CV Gross Sales (float)
     - CV Net Sales (float)
     - CV COGS (float)
     - CV Gross Profit (float)

2. Model Training:
   - Select model type
   - Configure hyperparameters
   - Start training

3. Generate Forecasts:
   - Select trained model
   - Set forecast horizon
   - Generate and visualize predictions
   - Download forecast in Excel format

## Development

1. Adding New Models:
   - Inherit from base model class
   - Implement required methods
   - Register in model registry

2. Customizing Training:
   - Modify training pipeline
   - Add new hyperparameters
   - Configure Azure ML settings

3. Extending Web Interface:
   - Add new React components
   - Update API endpoints
   - Enhance visualizations

## License

Proprietary - All rights reserved
