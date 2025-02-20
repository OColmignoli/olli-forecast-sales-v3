"""
FastAPI backend for sales forecasting web interface.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="OLLI Sales Forecasting",
    description="Advanced sales forecasting system using multiple ML models",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://victorious-ground-072a83c1e-preview.westus2.6.azurestaticapps.net"],
    allow_credentials=False,  # Changed to False since we're not using cookies
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
    expose_headers=["*"]
)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/models/metrics")
def get_model_metrics():
    return {"metrics": {"accuracy": 0.95}}

@app.get("/api/forecast/generate")
def generate_forecast(horizon: int = 52):
    return {"forecast": [100.0] * horizon}
