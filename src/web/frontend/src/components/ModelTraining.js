import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Grid,
  Chip,
} from '@mui/material';
import {
  Timeline,
  Memory,
  Psychology,
  ShowChart,
} from '@mui/icons-material';
import axios from 'axios';

const models = [
  {
    id: 'lstm',
    name: 'LSTM',
    description: 'Long Short-Term Memory Network',
    icon: <Memory />,
    color: '#1976d2',
  },
  {
    id: 'transformer',
    name: 'Transformer',
    description: 'Temporal Fusion Transformer',
    icon: <Psychology />,
    color: '#2196f3',
  },
  {
    id: 'deepar',
    name: 'DeepAR+',
    description: 'Deep Auto-Regressive Model',
    icon: <Timeline />,
    color: '#03a9f4',
  },
  {
    id: 'cnn',
    name: 'CNN',
    description: 'Convolutional Neural Network',
    icon: <Memory />,
    color: '#00bcd4',
  },
  {
    id: 'prophet',
    name: 'Prophet',
    description: 'Facebook Prophet Model',
    icon: <ShowChart />,
    color: '#009688',
  },
];

function ModelTraining() {
  const [selectedModels, setSelectedModels] = useState([]);
  const [training, setTraining] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [metrics, setMetrics] = useState({});

  const handleModelSelect = (modelId) => {
    if (selectedModels.includes(modelId)) {
      setSelectedModels(selectedModels.filter(id => id !== modelId));
    } else {
      setSelectedModels([...selectedModels, modelId]);
    }
  };

  const handleTraining = async () => {
    if (selectedModels.length === 0) return;

    setTraining(true);
    setError(null);
    setSuccess(false);

    try {
      const response = await axios.post('/api/models/train', {
        model_names: selectedModels,
      });

      setMetrics(response.data);
      setSuccess(true);
    } catch (err) {
      setError(err.response?.data?.detail || 'Training failed');
    } finally {
      setTraining(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Model Training
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        {models.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <Card
              sx={{
                cursor: 'pointer',
                transition: 'transform 0.2s',
                '&:hover': {
                  transform: 'translateY(-4px)',
                },
                border: selectedModels.includes(model.id)
                  ? `2px solid ${model.color}`
                  : 'none',
              }}
              onClick={() => handleModelSelect(model.id)}
            >
              <CardContent>
                <Box
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    mb: 2,
                  }}
                >
                  <Box
                    sx={{
                      backgroundColor: model.color,
                      borderRadius: '50%',
                      p: 1,
                      mr: 2,
                      color: 'white',
                    }}
                  >
                    {model.icon}
                  </Box>
                  <Box>
                    <Typography variant="h6">{model.name}</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {model.description}
                    </Typography>
                  </Box>
                </Box>

                {metrics[model.id] && (
                  <Box sx={{ mt: 2 }}>
                    <Chip
                      label={`RMSE: ${metrics[model.id].rmse.toFixed(4)}`}
                      size="small"
                      color="primary"
                      sx={{ mr: 1 }}
                    />
                    <Chip
                      label={`MAE: ${metrics[model.id].mae.toFixed(4)}`}
                      size="small"
                      color="secondary"
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Card>
        <CardContent>
          <Box sx={{ mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Selected Models
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {selectedModels.map((modelId) => {
                const model = models.find(m => m.id === modelId);
                return (
                  <Chip
                    key={modelId}
                    label={model.name}
                    onDelete={() => handleModelSelect(modelId)}
                    color="primary"
                    icon={model.icon}
                  />
                );
              })}
            </Box>
          </Box>

          <Button
            variant="contained"
            onClick={handleTraining}
            disabled={training || selectedModels.length === 0}
            startIcon={<Science />}
            sx={{ mr: 2 }}
          >
            Train Models
          </Button>

          {training && (
            <Box sx={{ mt: 2 }}>
              <LinearProgress />
            </Box>
          )}

          {error && (
            <Alert severity="error" sx={{ mt: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert
              icon={<CheckCircle />}
              severity="success"
              sx={{ mt: 2 }}
            >
              Models trained successfully!
            </Alert>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

export default ModelTraining;
