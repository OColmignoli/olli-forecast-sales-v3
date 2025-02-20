param location string = 'westus2'
param projectName string = 'olli-forecast-sales'
param environment string = 'prod'

// Variables
var webAppName = '${projectName}-${environment}'
var appServicePlanName = '${projectName}-plan-${environment}'
var staticWebAppName = '${projectName}-static-${environment}'
var keyVaultName = take(replace(replace('${projectName}${environment}kv', '-', ''), '_', ''), 24)
var appInsightsName = '${projectName}-insights-${environment}'

// App Service Plan
resource appServicePlan 'Microsoft.Web/serverfarms@2021-03-01' = {
  name: appServicePlanName
  location: location
  sku: {
    name: 'P1v2'
    tier: 'PremiumV2'
  }
  properties: {
    reserved: true
  }
}

// Backend Web App
resource webApp 'Microsoft.Web/sites@2021-03-01' = {
  name: '${webAppName}-api'
  location: location
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      pythonVersion: '3.9'
      linuxFxVersion: 'PYTHON|3.9'
      appSettings: [
        {
          name: 'AZURE_SUBSCRIPTION_ID'
          value: 'c828c783-7a28-48f4-b56f-a6c189437d77'
        }
        {
          name: 'AZURE_RESOURCE_GROUP'
          value: 'OLLI-resource'
        }
        {
          name: 'AZURE_WORKSPACE_NAME'
          value: 'OLLI_ML_Forecast'
        }
        {
          name: 'AZURE_REGION'
          value: 'westus2'
        }
        {
          name: 'APPINSIGHTS_INSTRUMENTATIONKEY'
          value: appInsights.properties.InstrumentationKey
        }
        {
          name: 'WEBSITE_MOUNT_ENABLED'
          value: '1'
        }
      ]
    }
  }
  identity: {
    type: 'SystemAssigned'
  }
}

// Static Web App for Frontend
resource staticWebApp 'Microsoft.Web/staticSites@2021-03-01' = {
  name: staticWebAppName
  location: location
  sku: {
    name: 'Standard'
    tier: 'Standard'
  }
  properties: {
    repositoryUrl: 'https://github.com/OLLI/olli-forecast-sales-v3'
    branch: 'main'
    buildProperties: {
      appLocation: '/src/web/frontend'
      apiLocation: '/api'
      outputLocation: 'build'
    }
  }
}

// Key Vault
resource keyVault 'Microsoft.KeyVault/vaults@2021-06-01-preview' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    accessPolicies: [
      {
        tenantId: subscription().tenantId
        objectId: webApp.identity.principalId
        permissions: {
          secrets: [
            'get'
            'list'
          ]
        }
      }
    ]
  }
}

// Application Insights
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: appInsightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Outputs
output webAppUrl string = webApp.properties.defaultHostName
output staticWebAppUrl string = staticWebApp.properties.defaultHostname
output keyVaultName string = keyVault.name
