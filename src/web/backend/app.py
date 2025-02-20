"""Backend API for sales forecasting application."""
import os
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from azure.storage.blob import BlobServiceClient
import mlflow
from mlflow.tracking import MlflowClient

from models.stacking_model import StackingModel
from models.config import StackingConfig
from utils.data_preprocessing import validate_columns, prepare_data_for_training

app = Flask(__name__)

# Configure Azure Storage
AZURE_STORAGE_CONNECTION_STRING = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
CONTAINER_NAME = 'sales-data'
blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Configure MLflow
mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
mlflow_client = MlflowClient()

# Training jobs status storage
training_jobs = {}

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Handle sales data file upload."""
    if 'file' not in request.files:
        return jsonify({'message': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'message': 'Only CSV files are supported'}), 400
    
    try:
        # Read and validate the data
        df = pd.read_csv(file)
        if not validate_columns(df):
            return jsonify({'message': 'Invalid file format'}), 400
        
        # Upload to Azure Blob Storage
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        blob_name = f'uploads/{timestamp}_{filename}'
        
        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(file.read())
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': blob_name
        })
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/models/train', methods=['POST'])
def train_models():
    """Start model training process."""
    data = request.json
    if not data or 'models' not in data:
        return jsonify({'message': 'No models specified'}), 400
    
    try:
        # Get latest data from blob storage
        blobs = list(container_client.list_blobs(name_starts_with='uploads/'))
        latest_blob = max(blobs, key=lambda b: b.last_modified)
        
        blob_client = container_client.get_blob_client(latest_blob.name)
        data_stream = blob_client.download_blob()
        df = pd.read_csv(data_stream)
        
        # Prepare data
        X, y = prepare_data_for_training(df)
        
        # Create and train stacking model
        config = StackingConfig(
            base_models=data['models'],
            meta_model=data.get('meta_model', 'xgboost')
        )
        
        model = StackingModel(config)
        
        # Start training in a separate thread
        from threading import Thread
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        training_jobs[job_id] = {
            'status': 'running',
            'progress': 0,
            'error': None
        }
        
        def train_async():
            try:
                # Train model
                model.train(df)
                training_jobs[job_id]['status'] = 'completed'
                training_jobs[job_id]['progress'] = 100
            except Exception as e:
                training_jobs[job_id]['status'] = 'failed'
                training_jobs[job_id]['error'] = str(e)
        
        Thread(target=train_async).start()
        
        return jsonify({
            'message': 'Training started',
            'job_id': job_id
        })
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/models/training/status/<job_id>', methods=['GET'])
def get_training_status(job_id):
    """Get status of a training job."""
    if job_id not in training_jobs:
        return jsonify({'message': 'Job not found'}), 404
    
    job = training_jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'error': job['error']
    })

@app.route('/api/forecast/latest', methods=['GET'])
def get_latest_forecast():
    """Get latest forecast results."""
    try:
        # Get latest model from MLflow
        latest_run = mlflow_client.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name('sales_forecast').experiment_id],
            order_by=['start_time DESC'],
            max_results=1
        )[0]
        
        model_uri = f"runs:/{latest_run.info.run_id}/model"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Get latest data
        blobs = list(container_client.list_blobs(name_starts_with='uploads/'))
        latest_blob = max(blobs, key=lambda b: b.last_modified)
        
        blob_client = container_client.get_blob_client(latest_blob.name)
        data_stream = blob_client.download_blob()
        df = pd.read_csv(data_stream)
        
        # Generate forecast
        forecast = model.predict(df)
        
        # Format response
        result = []
        for i, (_, row) in enumerate(df.iterrows()):
            point = {
                'date': row['date'],
                'actual': {
                    'Volume': row['Volume'],
                    'CV Gross Sales': row['CV Gross Sales'],
                    'CV Net Sales': row['CV Net Sales'],
                    'CV COGS': row['CV COGS'],
                    'CV Gross Profit': row['CV Gross Profit']
                }
            }
            if i < len(forecast):
                point['forecast'] = {
                    'Volume': forecast[i][0],
                    'CV Gross Sales': forecast[i][1],
                    'CV Net Sales': forecast[i][2],
                    'CV COGS': forecast[i][3],
                    'CV Gross Profit': forecast[i][4]
                }
            result.append(point)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

@app.route('/api/forecast/download', methods=['GET'])
def download_forecast():
    """Download forecast results as Excel file."""
    try:
        # Get latest forecast
        response = get_latest_forecast()
        forecast_data = response.json
        
        # Create Excel file
        df = pd.DataFrame(forecast_data)
        
        # Save to temporary file
        temp_file = 'temp_forecast.xlsx'
        df.to_excel(temp_file, index=False)
        
        return send_file(
            temp_file,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='forecast_results.xlsx'
        )
    
    except Exception as e:
        return jsonify({'message': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
