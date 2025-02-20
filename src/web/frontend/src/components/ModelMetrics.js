import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from 'recharts';
import axios from 'axios';

function ModelMetrics() {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('/api/models/metrics');
      setMetrics(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Box
        sx={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          minHeight: 400,
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  const prepareChartData = (metricName) => {
    return Object.entries(metrics).map(([model, modelMetrics]) => ({
      model: model.charAt(0).toUpperCase() + model.slice(1),
      value: modelMetrics[metricName] || 0,
    }));
  };

  const prepareRadarData = () => {
    const metricNames = ['rmse', 'mae', 'mape'];
    return Object.entries(metrics).map(([model, modelMetrics]) => ({
      model: model.charAt(0).toUpperCase() + model.slice(1),
      ...metricNames.reduce((acc, metric) => ({
        ...acc,
        [metric]: modelMetrics[metric] || 0,
      }), {}),
    }));
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Model Performance Metrics
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                RMSE Comparison
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={prepareChartData('rmse')}
                    margin={{
                      top: 5,
                      right: 30,
                      left: 20,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar
                      dataKey="value"
                      name="RMSE"
                      fill="#1976d2"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                MAE Comparison
              </Typography>
              <Box sx={{ height: 300 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={prepareChartData('mae')}
                    margin={{
                      top: 5,
                      right: 30,
                      left: 20,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="model" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar
                      dataKey="value"
                      name="MAE"
                      fill="#2196f3"
                    />
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Performance Overview
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={prepareRadarData()}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="model" />
                    <PolarRadiusAxis />
                    <Radar
                      name="RMSE"
                      dataKey="rmse"
                      stroke="#1976d2"
                      fill="#1976d2"
                      fillOpacity={0.6}
                    />
                    <Radar
                      name="MAE"
                      dataKey="mae"
                      stroke="#2196f3"
                      fill="#2196f3"
                      fillOpacity={0.6}
                    />
                    <Radar
                      name="MAPE"
                      dataKey="mape"
                      stroke="#03a9f4"
                      fill="#03a9f4"
                      fillOpacity={0.6}
                    />
                    <Legend />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default ModelMetrics;
