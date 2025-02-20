import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Grid,
} from '@mui/material';
import {
  Timeline,
  Download,
  CheckCircle,
  Error,
} from '@mui/icons-material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import axios from 'axios';

function ForecastGeneration() {
  const [modelName, setModelName] = useState('stacking');
  const [horizon, setHorizon] = useState(52);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [forecast, setForecast] = useState(null);
  const [downloading, setDownloading] = useState(false);

  const handleGenerate = async () => {
    setGenerating(true);
    setError(null);
    setForecast(null);

    try {
      const response = await axios.post('/api/forecast/generate', {
        model_name: modelName,
        horizon: horizon,
      });

      setForecast({
        filename: response.data.filename,
        data: response.data.forecast,
      });
    } catch (err) {
      setError(err.response?.data?.detail || 'Forecast generation failed');
    } finally {
      setGenerating(false);
    }
  };

  const handleDownload = async () => {
    if (!forecast) return;

    setDownloading(true);
    try {
      const response = await axios.get(
        `/api/forecast/download/${forecast.filename}`,
        {
          responseType: 'blob',
        }
      );

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute(
        'download',
        forecast.filename.replace('.csv', '.xlsx')
      );
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Download failed');
    } finally {
      setDownloading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Generate Forecast
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Forecast Settings
              </Typography>

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Model</InputLabel>
                <Select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  label="Model"
                >
                  <MenuItem value="stacking">Stacking Ensemble</MenuItem>
                  <MenuItem value="lstm">LSTM</MenuItem>
                  <MenuItem value="transformer">Transformer</MenuItem>
                  <MenuItem value="deepar">DeepAR+</MenuItem>
                  <MenuItem value="cnn">CNN</MenuItem>
                  <MenuItem value="prophet">Prophet</MenuItem>
                </Select>
              </FormControl>

              <TextField
                fullWidth
                type="number"
                label="Forecast Horizon (weeks)"
                value={horizon}
                onChange={(e) => setHorizon(parseInt(e.target.value))}
                sx={{ mb: 2 }}
              />

              <Button
                fullWidth
                variant="contained"
                onClick={handleGenerate}
                disabled={generating}
                startIcon={<Timeline />}
              >
                Generate Forecast
              </Button>

              {forecast && (
                <Button
                  fullWidth
                  variant="outlined"
                  onClick={handleDownload}
                  disabled={downloading}
                  startIcon={<Download />}
                  sx={{ mt: 2 }}
                >
                  Download Excel
                </Button>
              )}

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Forecast Visualization
              </Typography>

              {forecast ? (
                <Box sx={{ height: 400 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={forecast.data}
                      margin={{
                        top: 5,
                        right: 30,
                        left: 20,
                        bottom: 5,
                      }}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis
                        dataKey="forecast_date"
                        tickFormatter={(date) => new Date(date).toLocaleDateString()}
                      />
                      <YAxis />
                      <Tooltip
                        labelFormatter={(date) => new Date(date).toLocaleDateString()}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="forecast_value"
                        name="Forecast"
                        stroke="#1976d2"
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              ) : (
                <Box
                  sx={{
                    height: 400,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Typography color="textSecondary">
                    Generate a forecast to see visualization
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ForecastGeneration;
