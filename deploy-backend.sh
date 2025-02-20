#!/bin/bash

# Azure configuration
SUBSCRIPTION_ID="c828c783-7a28-48f4-b56f-a6c189437d77"
RESOURCE_GROUP="OLLI-resource"
APP_NAME="olli-forecast-sales-prod-api"
LOCATION="westus2"
REGISTRY_NAME="olliforecastv3registry"

# Login to Azure (if not already logged in)
az account show || az login

# Set subscription
az account set --subscription $SUBSCRIPTION_ID

# Create App Service plan if it doesn't exist
az appservice plan create \
    --name "${APP_NAME}-plan" \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku B1 \
    --is-linux

# Create web app if it doesn't exist
az webapp create \
    --resource-group $RESOURCE_GROUP \
    --plan "${APP_NAME}-plan" \
    --name $APP_NAME \
    --runtime "PYTHON:3.9"

# Configure web app settings
az webapp config set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --startup-file "uvicorn src.web.api:app --host 0.0.0.0 --port 8000"

# Enable system-assigned managed identity
az webapp identity assign \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME

# Configure CORS
az webapp cors add \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --allowed-origins "https://victorious-ground-072a83c1e-preview.westus2.6.azurestaticapps.net"

# Get ACR credentials
ACR_USERNAME=$(az acr credential show -n $REGISTRY_NAME --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n $REGISTRY_NAME --query "passwords[0].value" -o tsv)

# Build and push Docker image
az acr build \
    --registry $REGISTRY_NAME \
    --image "${APP_NAME}:latest" \
    --file Dockerfile.backend .

# Update web app with container settings
az webapp config container set \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --docker-custom-image-name "${REGISTRY_NAME}.azurecr.io/${APP_NAME}:latest" \
    --docker-registry-server-url "https://${REGISTRY_NAME}.azurecr.io" \
    --docker-registry-server-user $ACR_USERNAME \
    --docker-registry-server-password $ACR_PASSWORD
