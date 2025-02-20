import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { Box } from '@mui/material';

// Components
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import DataUpload from './components/DataUpload';
import ModelTraining from './components/ModelTraining';
import ForecastGeneration from './components/ForecastGeneration';
import ModelMetrics from './components/ModelMetrics';

// OLLI theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2', // OLLI blue
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e', // OLLI red
      light: '#ff4081',
      dark: '#9a0036',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Header />
          <Sidebar />
          <Box
            component="main"
            sx={{
              flexGrow: 1,
              p: 3,
              mt: 8,
              ml: { sm: 30 },
              backgroundColor: 'background.default',
            }}
          >
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/upload" element={<DataUpload />} />
              <Route path="/training" element={<ModelTraining />} />
              <Route path="/forecast" element={<ForecastGeneration />} />
              <Route path="/metrics" element={<ModelMetrics />} />
            </Routes>
          </Box>
        </Box>
      </Router>
    </ThemeProvider>
  );
}

export default App;
