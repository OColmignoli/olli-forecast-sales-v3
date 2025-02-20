import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Checkbox,
  CircularProgress,
  FormControl,
  FormControlLabel,
  FormGroup,
  InputLabel,
  MenuItem,
  Paper,
  Select,
  Typography,
  Alert,
  LinearProgress
} from '@mui/material';
import axios from 'axios';

const ModelTrainingForm = () => {
  const [selectedModels, setSelectedModels] = useState([]);
  const [metaModel, setMetaModel] = useState('xgboost');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');

  const availableModels = [
    { id: 'lstm', name: 'LSTM' },
    { id: 'transformer', name: 'Transformer' },
    { id: 'deepar', name: 'DeepAR+' },
    { id: 'cnn', name: 'CNN' },
    { id: 'prophet', name: 'Prophet' }
  ];

  const handleModelChange = (event) => {
    const model = event.target.name;
    const isChecked = event.target.checked;

    setSelectedModels(prev => 
      isChecked 
        ? [...prev, model]
        : prev.filter(m => m !== model)
    );
  };

  const handleMetaModelChange = (event) => {
    setMetaModel(event.target.value);
  };

  const checkTrainingStatus = async (jobId) => {
    try {
      const response = await axios.get(`/api/models/training/status/${jobId}`);
      const { status, progress: currentProgress } = response.data;

      setStatus(status);
      setProgress(currentProgress);

      if (status === 'completed') {
        setSuccess(true);
        setLoading(false);
      } else if (status === 'failed') {
        setError('Training failed. Please try again.');
        setLoading(false);
      } else {
        // Continue polling
        setTimeout(() => checkTrainingStatus(jobId), 5000);
      }
    } catch (err) {
      setError('Error checking training status');
      setLoading(false);
    }
  };

  const handleTraining = async () => {
    if (selectedModels.length === 0) {
      setError('Please select at least one model');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(false);
    setProgress(0);
    setStatus('');

    try {
      const response = await axios.post('/api/models/train', {
        models: selectedModels,
        meta_model: metaModel
      });

      const { job_id } = response.data;
      checkTrainingStatus(job_id);
    } catch (err) {
      setError(err.response?.data?.message || 'Error starting training');
      setLoading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Train Models
      </Typography>

      <Paper sx={{ p: 3 }}>
        <FormGroup sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Select Base Models
          </Typography>
          {availableModels.map((model) => (
            <FormControlLabel
              key={model.id}
              control={
                <Checkbox
                  checked={selectedModels.includes(model.id)}
                  onChange={handleModelChange}
                  name={model.id}
                />
              }
              label={model.name}
            />
          ))}
        </FormGroup>

        <FormControl fullWidth sx={{ mb: 3 }}>
          <InputLabel>Meta Model</InputLabel>
          <Select
            value={metaModel}
            label="Meta Model"
            onChange={handleMetaModelChange}
          >
            <MenuItem value="xgboost">XGBoost</MenuItem>
            <MenuItem value="randomforest">Random Forest</MenuItem>
          </Select>
        </FormControl>

        {loading && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" gutterBottom>
              {status || 'Preparing training...'}
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={progress} 
              sx={{ mb: 1 }}
            />
            <Typography variant="body2" color="text.secondary">
              {`${Math.round(progress)}%`}
            </Typography>
          </Box>
        )}

        <Button
          variant="contained"
          color="primary"
          onClick={handleTraining}
          disabled={loading}
          sx={{ mt: 2 }}
        >
          {loading ? <CircularProgress size={24} /> : 'Start Training'}
        </Button>

        {error && (
          <Alert severity="error" sx={{ mt: 2 }}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mt: 2 }}>
            Models trained successfully!
          </Alert>
        )}
      </Paper>
    </Box>
  );
};

export default ModelTrainingForm;
