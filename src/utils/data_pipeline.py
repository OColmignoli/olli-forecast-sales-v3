"""
Data processing pipeline for sales forecasting.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from azure.storage.blob import BlobServiceClient
import logging
from datetime import datetime
import holidays
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

class DataPipeline:
    """Handles data processing and feature engineering."""
    
    def __init__(
        self,
        blob_client: BlobServiceClient,
        container_name: str = "sales-data",
        scaler: Optional[StandardScaler] = None
    ):
        """Initialize data pipeline."""
        self.blob_client = blob_client
        self.container_name = container_name
        self.scaler = scaler or StandardScaler()
        self.us_holidays = holidays.US()
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate input data format and content."""
        try:
            errors = []
            
            # Check required columns
            required_columns = [
                'Transaction Year',
                'Transaction Week',
                'Volume',
                'CV Gross Sales',
                'CV Net Sales',
                'CV COGS',
                'CV Gross Profit'
            ]
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
            
            # Check data types
            if not errors:
                # Year validation
                if not pd.to_numeric(data['Transaction Year'], errors='coerce').notnull().all():
                    errors.append("Transaction Year must be numeric")
                else:
                    invalid_years = data[
                        (data['Transaction Year'] < 2000) | 
                        (data['Transaction Year'] > datetime.now().year)
                    ]
                    if not invalid_years.empty:
                        errors.append("Invalid Transaction Year values found")
                
                # Week validation
                if not pd.to_numeric(data['Transaction Week'], errors='coerce').notnull().all():
                    errors.append("Transaction Week must be numeric")
                else:
                    invalid_weeks = data[
                        (data['Transaction Week'] < 1) | 
                        (data['Transaction Week'] > 53)
                    ]
                    if not invalid_weeks.empty:
                        errors.append("Invalid Transaction Week values found")
                
                # Numeric columns validation
                numeric_columns = [
                    'Volume',
                    'CV Gross Sales',
                    'CV Net Sales',
                    'CV COGS',
                    'CV Gross Profit'
                ]
                
                for col in numeric_columns:
                    if not pd.to_numeric(data[col], errors='coerce').notnull().all():
                        errors.append(f"{col} must be numeric")
            
            return len(errors) == 0, errors
        
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data."""
        try:
            # Create copy to avoid modifying original data
            df = data.copy()
            
            # Sort by year and week
            df = df.sort_values(['Transaction Year', 'Transaction Week'])
            
            # Create datetime index
            df['date'] = df.apply(
                lambda x: datetime(
                    int(x['Transaction Year']),
                    1,
                    1
                ) + pd.Timedelta(weeks=int(x['Transaction Week'])-1),
                axis=1
            )
            
            # Set date as index
            df.set_index('date', inplace=True)
            
            return df
        
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        try:
            df = data.copy()
            
            # Time-based features
            df['year'] = df.index.year
            df['month'] = df.index.month
            df['week_of_year'] = df.index.isocalendar().week
            df['day_of_week'] = df.index.dayofweek
            
            # Holiday features
            df['is_holiday'] = df.index.map(lambda x: x in self.us_holidays)
            df['days_to_holiday'] = df.index.map(
                lambda x: min(
                    [(h - x).days for h in self.us_holidays.keys() if (h - x).days > 0],
                    default=0
                )
            )
            
            # Lag features
            for lag in [1, 2, 4, 8, 12, 26, 52]:
                df[f'sales_lag_{lag}'] = df['CV Gross Sales'].shift(lag)
                df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            
            # Rolling statistics
            for window in [4, 12, 26]:
                df[f'sales_rolling_mean_{window}'] = df['CV Gross Sales'].rolling(window).mean()
                df[f'sales_rolling_std_{window}'] = df['CV Gross Sales'].rolling(window).std()
                df[f'volume_rolling_mean_{window}'] = df['Volume'].rolling(window).mean()
            
            # Ratios and derived features
            df['profit_margin'] = df['CV Gross Profit'] / df['CV Gross Sales']
            df['sales_per_volume'] = df['CV Gross Sales'] / df['Volume']
            
            # Drop rows with NaN values from lag features
            df = df.dropna()
            
            return df
        
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            raise
    
    def scale_features(
        self,
        data: pd.DataFrame,
        fit: bool = True
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """Scale numerical features."""
        try:
            # Select numerical columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            
            if fit:
                # Fit and transform
                scaled_data = self.scaler.fit_transform(data[numeric_cols])
            else:
                # Transform only
                scaled_data = self.scaler.transform(data[numeric_cols])
            
            # Create DataFrame with scaled values
            scaled_df = pd.DataFrame(
                scaled_data,
                columns=numeric_cols,
                index=data.index
            )
            
            # Add non-numeric columns back
            for col in data.columns:
                if col not in numeric_cols:
                    scaled_df[col] = data[col]
            
            return scaled_df
        
        except Exception as e:
            logger.error(f"Feature scaling failed: {str(e)}")
            raise
    
    def save_to_blob(
        self,
        data: pd.DataFrame,
        filename: str,
        container: Optional[str] = None
    ):
        """Save data to Azure Blob Storage."""
        try:
            # Get container client
            container_client = self.blob_client.get_container_client(
                container or self.container_name
            )
            
            # Convert to CSV
            csv_data = data.to_csv()
            
            # Upload to blob
            blob_client = container_client.get_blob_client(filename)
            blob_client.upload_blob(csv_data, overwrite=True)
            
            logger.info(f"Successfully saved data to {filename}")
        
        except Exception as e:
            logger.error(f"Failed to save data to blob: {str(e)}")
            raise
    
    def load_from_blob(
        self,
        filename: str,
        container: Optional[str] = None
    ) -> pd.DataFrame:
        """Load data from Azure Blob Storage."""
        try:
            # Get container client
            container_client = self.blob_client.get_container_client(
                container or self.container_name
            )
            
            # Download blob
            blob_client = container_client.get_blob_client(filename)
            data = blob_client.download_blob().readall()
            
            # Convert to DataFrame
            df = pd.read_csv(data)
            
            logger.info(f"Successfully loaded data from {filename}")
            return df
        
        except Exception as e:
            logger.error(f"Failed to load data from blob: {str(e)}")
            raise
    
    def process_weekly_update(
        self,
        new_data: pd.DataFrame,
        update_models: bool = True
    ) -> Dict[str, Any]:
        """Process weekly data update."""
        try:
            # Validate new data
            is_valid, errors = self.validate_data(new_data)
            if not is_valid:
                raise ValueError(f"Invalid data: {errors}")
            
            # Load existing data
            existing_data = self.load_from_blob("processed_data.csv", "processed-data")
            
            # Preprocess new data
            processed_new_data = self.preprocess_data(new_data)
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, processed_new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            
            # Engineer features
            featured_data = self.engineer_features(combined_data)
            
            # Scale features
            scaled_data = self.scale_features(featured_data, fit=True)[0]
            
            # Save processed data
            self.save_to_blob(scaled_data, "processed_data.csv", "processed-data")
            
            # Save feature scaler
            scaler_data = {
                'mean_': self.scaler.mean_.tolist(),
                'scale_': self.scaler.scale_.tolist(),
                'var_': self.scaler.var_.tolist(),
                'n_samples_seen_': int(self.scaler.n_samples_seen_)
            }
            
            self.save_to_blob(
                pd.DataFrame([scaler_data]),
                "feature_scaler.json",
                "processed-data"
            )
            
            return {
                "status": "success",
                "new_records": len(new_data),
                "total_records": len(scaled_data),
                "update_date": datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Weekly update failed: {str(e)}")
            raise
