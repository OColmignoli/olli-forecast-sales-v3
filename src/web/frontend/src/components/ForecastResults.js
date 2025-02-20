import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Paper,
  Typography,
  Alert,
  FormControl,
  InputLabel,
  MenuItem,
  Select
} from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Download } from '@mui/icons-material';
import axios from 'axios';

const ForecastResults = () => {
  const [forecasts, setForecasts] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('CV Net Sales');
  const [downloadLoading, setDownloadLoading] = useState(false);

  const metrics = [
    'Volume',
    'CV Gross Sales',
    'CV Net Sales',
    'CV COGS',
    'CV Gross Profit'
  ];

  useEffect(() => {
    fetchForecasts();
  }, []);

  const fetchForecasts = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get('/api/forecast/latest');
      setForecasts(response.data);
    } catch (err) {
      setError(err.response?.data?.message || 'Error fetching forecasts');
    } finally {
      setLoading(false);
    }
  };

  const handleMetricChange = (event) => {
    setSelectedMetric(event.target.value);
  };

  const handleDownload = async () => {
    setDownloadLoading(true);
    try {
      const response = await axios.get('/api/forecast/download', {
        responseType: 'blob'
      });

      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'forecast_results.xlsx');
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Error downloading forecast results');
    } finally {
      setDownloadLoading(false);
    }
  };

  const formatData = (forecasts) => {
    if (!forecasts) return [];

    return forecasts.map((point) => ({
      date: new Date(point.date).toLocaleDateString(),
      actual: point.actual?.[selectedMetric],
      forecast: point.forecast?.[selectedMetric],
      lower_bound: point.bounds?.lower?.[selectedMetric],
      upper_bound: point.bounds?.upper?.[selectedMetric]
    }));
  };

  return (
    <Box sx={{ maxWidth: 1200, mx: 'auto', p: 3 }}>
      <Typography variant="h5" gutterBottom>
        Forecast Results
      </Typography>

      <Paper sx={{ p: 3 }}>
        <Box sx={{ mb: 3, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Metric</InputLabel>
            <Select
              value={selectedMetric}
              label="Metric"
              onChange={handleMetricChange}
            >
              {metrics.map((metric) => (
                <MenuItem key={metric} value={metric}>
                  {metric}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Button
            variant="contained"
            color="primary"
            startIcon={<Download />}
            onClick={handleDownload}
            disabled={downloadLoading || !forecasts}
          >
            {downloadLoading ? <CircularProgress size={24} /> : 'Download Excel'}
          </Button>
        </Box>

        {loading ? (
          <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
            <CircularProgress />
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        ) : forecasts ? (
          <Box sx={{ height: 400 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={formatData(forecasts)}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#8884d8"
                  name="Actual"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#82ca9d"
                  name="Forecast"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="lower_bound"
                  stroke="#ff7300"
                  name="Lower Bound"
                  strokeDasharray="3 3"
                />
                <Line
                  type="monotone"
                  dataKey="upper_bound"
                  stroke="#ff7300"
                  name="Upper Bound"
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        ) : (
          <Typography variant="body1" sx={{ textAlign: 'center', p: 3 }}>
            No forecast data available
          </Typography>
        )}
      </Paper>
    </Box>
  );
};

export default ForecastResults;
