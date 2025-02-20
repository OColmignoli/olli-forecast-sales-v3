"""Test suite for API endpoints."""
import pytest
import json
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from io import BytesIO
from datetime import datetime

from src.web.backend.app import app

@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_csv():
    """Create sample CSV file."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='W-MON')
    df = pd.DataFrame({
        'Transaction Year': dates.year,
        'Transaction Week': dates.isocalendar().week,
        'Volume': np.random.normal(1000, 100, 100),
        'CV Gross Sales': np.random.normal(5000, 500, 100),
        'CV Net Sales': np.random.normal(4000, 400, 100),
        'CV COGS': np.random.normal(3000, 300, 100),
        'CV Gross Profit': np.random.normal(1000, 100, 100)
    })
    
    csv_file = BytesIO()
    df.to_csv(csv_file, index=False)
    csv_file.seek(0)
    return csv_file

@pytest.fixture
def mock_blob_service():
    """Create mock Azure Blob service."""
    with patch('azure.storage.blob.BlobServiceClient') as mock:
        container_client = Mock()
        blob_client = Mock()
        container_client.get_blob_client.return_value = blob_client
        mock.return_value.get_container_client.return_value = container_client
        yield mock

@pytest.fixture
def mock_mlflow():
    """Create mock MLflow client."""
    with patch('mlflow.tracking.MlflowClient') as mock:
        yield mock

def test_upload_data_success(client, sample_csv, mock_blob_service):
    """Test successful data upload."""
    response = client.post(
        '/api/data/upload',
        data={'file': (sample_csv, 'data.csv')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'filename' in data
    assert data['message'] == 'File uploaded successfully'

def test_upload_data_no_file(client):
    """Test data upload with no file."""
    response = client.post('/api/data/upload')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['message'] == 'No file provided'

def test_upload_data_invalid_format(client):
    """Test data upload with invalid format."""
    invalid_file = BytesIO(b'invalid data')
    response = client.post(
        '/api/data/upload',
        data={'file': (invalid_file, 'data.txt')},
        content_type='multipart/form-data'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'message' in data

def test_train_models_success(client, mock_blob_service, mock_mlflow):
    """Test successful model training."""
    data = {
        'models': ['lstm', 'transformer', 'prophet'],
        'meta_model': 'xgboost'
    }
    
    response = client.post(
        '/api/models/train',
        json=data
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'message' in data
    assert 'job_id' in data
    assert data['message'] == 'Training started'

def test_train_models_no_models(client):
    """Test model training with no models specified."""
    response = client.post(
        '/api/models/train',
        json={}
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['message'] == 'No models specified'

def test_get_training_status_success(client):
    """Test getting training status."""
    # Add a mock job
    job_id = '20200101_123456'
    app.training_jobs[job_id] = {
        'status': 'running',
        'progress': 50,
        'error': None
    }
    
    response = client.get(f'/api/models/training/status/{job_id}')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data['status'] == 'running'
    assert data['progress'] == 50
    assert data['error'] is None

def test_get_training_status_not_found(client):
    """Test getting status of non-existent job."""
    response = client.get('/api/models/training/status/invalid_id')
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['message'] == 'Job not found'

@patch('src.web.backend.app.mlflow.pyfunc.load_model')
def test_get_latest_forecast_success(mock_load_model, client, mock_mlflow):
    """Test getting latest forecast."""
    # Mock MLflow run
    mock_run = Mock()
    mock_run.info.run_id = 'test_run'
    mock_mlflow.search_runs.return_value = [mock_run]
    
    # Mock model predictions
    mock_model = Mock()
    mock_model.predict.return_value = np.random.normal(size=(10, 5))
    mock_load_model.return_value = mock_model
    
    response = client.get('/api/forecast/latest')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    assert isinstance(data, list)
    assert len(data) > 0
    assert 'date' in data[0]
    assert 'actual' in data[0]
    assert 'forecast' in data[0]

@patch('pandas.DataFrame.to_excel')
def test_download_forecast_success(mock_to_excel, client):
    """Test downloading forecast results."""
    response = client.get('/api/forecast/download')
    
    assert response.status_code == 200
    assert response.headers['Content-Type'] == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    assert 'attachment' in response.headers['Content-Disposition']
    assert 'forecast_results.xlsx' in response.headers['Content-Disposition']
