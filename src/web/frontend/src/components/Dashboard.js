import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  Timeline,
  TrendingUp,
  Assessment,
  Speed,
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
  AreaChart,
  Area,
} from 'recharts';
import axios from 'axios';

function Dashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      // Fetch latest metrics and forecasts
      const config = {
        withCredentials: false,
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Origin': 'https://victorious-ground-072a83c1e-preview.westus2.6.azurestaticapps.net'
        }
      };

      // First check if API is healthy
      // Update to match the new frontend URL
      const baseUrl = window.location.hostname.includes('localhost')
        ? 'http://localhost:8000'
        : 'https://olli-forecast-sales-prod-api.azurewebsites.net';
      const healthCheck = await axios.get(`${baseUrl}/health`, config);
      console.log('Health check response:', healthCheck.data);
      
      // Add delay to ensure health check is processed
      await new Promise(resolve => setTimeout(resolve, 1000));

      const [metricsResponse, forecastResponse] = await Promise.all([
        axios.get(`${baseUrl}/api/models/metrics`, config),
        axios.get(`${baseUrl}/api/forecast/generate`, {
          ...config,
          params: { horizon: 52 }
        }),
      ]);

      setData({
        metrics: metricsResponse.data,
        forecast: forecastResponse.data,
      });
    } catch (err) {
      console.error('Dashboard data fetch error:', err);
      if (err.response?.status === 404) {
        setError('API endpoint not found. Please check the server configuration.');
      } else if (err.response?.status === 403) {
        setError('Access denied. Please check your authentication.');
      } else if (err.code === 'ERR_NETWORK') {
        setError('Network error. Please check your connection and try again.');
      } else {
        setError(err.response?.data?.detail || err.message || 'Failed to load dashboard data');
      }
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

  const getMetricColor = (value) => {
    if (value < 0.1) return '#4caf50';
    if (value < 0.2) return '#ff9800';
    return '#f44336';
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Sales Forecasting Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Summary Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    backgroundColor: 'primary.main',
                    borderRadius: '50%',
                    p: 1,
                    mr: 2,
                    color: 'white',
                  }}
                >
                  <Timeline />
                </Box>
                <Typography variant="h6">
                  Forecast Accuracy
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {data?.metrics?.stacking?.accuracy ? (data.metrics.stacking.accuracy * 100).toFixed(1) : 'N/A'}%
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Last 4 weeks
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    backgroundColor: 'secondary.main',
                    borderRadius: '50%',
                    p: 1,
                    mr: 2,
                    color: 'white',
                  }}
                >
                  <TrendingUp />
                </Box>
                <Typography variant="h6">
                  Sales Trend
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {data?.metrics?.trend ? (
                  <>
                    {data.metrics.trend > 0 ? '+' : ''}
                    {(data.metrics.trend * 100).toFixed(1)}%
                  </>
                ) : 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                vs Previous Period
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    backgroundColor: '#4caf50',
                    borderRadius: '50%',
                    p: 1,
                    mr: 2,
                    color: 'white',
                  }}
                >
                  <Assessment />
                </Box>
                <Typography variant="h6">
                  Model Performance
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {data?.metrics?.stacking?.rmse ? data.metrics.stacking.rmse.toFixed(3) : 'N/A'}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Ensemble RMSE
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Box
                  sx={{
                    backgroundColor: '#ff9800',
                    borderRadius: '50%',
                    p: 1,
                    mr: 2,
                    color: 'white',
                  }}
                >
                  <Speed />
                </Box>
                <Typography variant="h6">
                  Model Health
                </Typography>
              </Box>
              <Typography variant="h4" sx={{ mb: 1 }}>
                {data.metrics.health_score.toFixed(1)}
              </Typography>
              <Typography variant="body2" color="textSecondary">
                Overall Score
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Charts */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Sales Forecast
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={data.forecast}
                    margin={{
                      top: 5,
                      right: 30,
                      left: 20,
                      bottom: 5,
                    }}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      tickFormatter={(date) => new Date(date).toLocaleDateString()}
                    />
                    <YAxis />
                    <Tooltip
                      labelFormatter={(date) => new Date(date).toLocaleDateString()}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual Sales"
                      stroke="#1976d2"
                      strokeWidth={2}
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast"
                      name="Forecast"
                      stroke="#2196f3"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Contribution
              </Typography>
              <Box sx={{ height: 400 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={Object.entries(data.metrics.model_weights).map(
                      ([model, weight]) => ({
                        model: model.charAt(0).toUpperCase() + model.slice(1),
                        weight: weight * 100,
                      })
                    )}
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
                    <Area
                      type="monotone"
                      dataKey="weight"
                      name="Contribution %"
                      stroke="#2196f3"
                      fill="#2196f3"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

export default Dashboard;
