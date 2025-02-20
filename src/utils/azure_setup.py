"""
Azure infrastructure setup and management.
"""
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    AmlCompute,
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment
)
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.storage.blob import BlobServiceClient
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class AzureInfraManager:
    """Manages Azure infrastructure setup and configuration."""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str,
        region: str,
        keyvault_name: str
    ):
        """Initialize Azure infrastructure manager."""
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.region = region
        self.keyvault_name = keyvault_name
        
        # Initialize Azure credentials
        self.credential = DefaultAzureCredential()
        
        # Initialize clients
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Azure clients."""
        try:
            # ML Workspace client
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name
            )
            
            # Key Vault client
            vault_url = f"https://{self.keyvault_name}.vault.azure.net/"
            self.keyvault_client = SecretClient(
                vault_url=vault_url,
                credential=self.credential
            )
            
            logger.info("Successfully initialized Azure clients")
        
        except Exception as e:
            logger.error(f"Failed to initialize Azure clients: {str(e)}")
            raise
    
    def setup_compute_cluster(
        self,
        cluster_name: str,
        vm_size: str = "Standard_NC6s_v3",
        min_instances: int = 0,
        max_instances: int = 4,
        idle_time_before_scale_down: int = 120
    ) -> AmlCompute:
        """Set up Azure ML compute cluster."""
        try:
            # Configure compute
            compute_config = AmlCompute(
                name=cluster_name,
                size=vm_size,
                min_instances=min_instances,
                max_instances=max_instances,
                idle_time_before_scale_down=idle_time_before_scale_down,
                tier="Dedicated"
            )
            
            # Create compute target
            compute = self.ml_client.compute.begin_create_or_update(compute_config).result()
            
            logger.info(f"Successfully created compute cluster: {cluster_name}")
            return compute
        
        except Exception as e:
            logger.error(f"Failed to create compute cluster: {str(e)}")
            raise
    
    def setup_online_endpoint(
        self,
        endpoint_name: str,
        auth_mode: str = "key"
    ) -> ManagedOnlineEndpoint:
        """Set up Azure ML online endpoint."""
        try:
            # Configure endpoint
            endpoint = ManagedOnlineEndpoint(
                name=endpoint_name,
                auth_mode=auth_mode,
                description="Sales forecasting endpoint"
            )
            
            # Create endpoint
            endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
            
            logger.info(f"Successfully created online endpoint: {endpoint_name}")
            return endpoint
        
        except Exception as e:
            logger.error(f"Failed to create online endpoint: {str(e)}")
            raise
    
    def setup_blob_storage(
        self,
        storage_account_name: str,
        container_name: str
    ) -> BlobServiceClient:
        """Set up Azure Blob Storage."""
        try:
            # Get storage account key from Key Vault
            storage_key = self.keyvault_client.get_secret(
                f"{storage_account_name}-key"
            ).value
            
            # Create Blob service client
            blob_service_client = BlobServiceClient(
                account_url=f"https://{storage_account_name}.blob.core.windows.net",
                credential=storage_key
            )
            
            # Create container if it doesn't exist
            container_client = blob_service_client.get_container_client(container_name)
            if not container_client.exists():
                container_client.create_container()
            
            logger.info(f"Successfully set up blob storage: {container_name}")
            return blob_service_client
        
        except Exception as e:
            logger.error(f"Failed to set up blob storage: {str(e)}")
            raise
    
    def store_secret(self, secret_name: str, secret_value: str):
        """Store secret in Azure Key Vault."""
        try:
            self.keyvault_client.set_secret(secret_name, secret_value)
            logger.info(f"Successfully stored secret: {secret_name}")
        
        except Exception as e:
            logger.error(f"Failed to store secret: {str(e)}")
            raise
    
    def get_secret(self, secret_name: str) -> str:
        """Get secret from Azure Key Vault."""
        try:
            secret = self.keyvault_client.get_secret(secret_name)
            return secret.value
        
        except Exception as e:
            logger.error(f"Failed to get secret: {str(e)}")
            raise
    
    def setup_all_infrastructure(self) -> Dict[str, Any]:
        """Set up all required Azure infrastructure."""
        try:
            # Set up compute clusters
            training_compute = self.setup_compute_cluster(
                "training-cluster",
                vm_size="Standard_NC6s_v3",
                max_instances=4
            )
            
            inference_compute = self.setup_compute_cluster(
                "inference-cluster",
                vm_size="Standard_DS3_v2",
                min_instances=1,
                max_instances=2
            )
            
            # Set up online endpoint
            endpoint = self.setup_online_endpoint("sales-forecast-endpoint")
            
            # Set up blob storage
            blob_client = self.setup_blob_storage(
                "ollisalesforecast",
                "sales-data"
            )
            
            # Create storage containers
            containers = ["raw-data", "processed-data", "models", "forecasts"]
            for container in containers:
                blob_client.create_container(container)
            
            infrastructure = {
                "training_compute": training_compute,
                "inference_compute": inference_compute,
                "endpoint": endpoint,
                "blob_client": blob_client
            }
            
            logger.info("Successfully set up all Azure infrastructure")
            return infrastructure
        
        except Exception as e:
            logger.error(f"Failed to set up infrastructure: {str(e)}")
            raise
