import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  CloudUpload as UploadIcon,
  Science as TrainingIcon,
  Timeline as ForecastIcon,
  Assessment as MetricsIcon,
} from '@mui/icons-material';

const drawerWidth = 240;

const menuItems = [
  {
    text: 'Dashboard',
    icon: <DashboardIcon />,
    path: '/',
  },
  {
    text: 'Upload Data',
    icon: <UploadIcon />,
    path: '/upload',
  },
  {
    text: 'Model Training',
    icon: <TrainingIcon />,
    path: '/training',
  },
  {
    text: 'Generate Forecast',
    icon: <ForecastIcon />,
    path: '/forecast',
  },
  {
    text: 'Model Metrics',
    icon: <MetricsIcon />,
    path: '/metrics',
  },
];

function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: 'background.paper',
          borderRight: '1px solid rgba(0, 0, 0, 0.12)',
        },
      }}
    >
      <Box sx={{ overflow: 'auto', mt: 8 }}>
        <List>
          {menuItems.map((item) => (
            <ListItem
              button
              key={item.text}
              onClick={() => navigate(item.path)}
              selected={location.pathname === item.path}
              sx={{
                mb: 1,
                mx: 1,
                borderRadius: 1,
                '&.Mui-selected': {
                  backgroundColor: 'primary.main',
                  color: 'white',
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                  '&:hover': {
                    backgroundColor: 'primary.dark',
                  },
                },
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: 40,
                  color: location.pathname === item.path ? 'white' : 'primary.main',
                }}
              >
                {item.icon}
              </ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItem>
          ))}
        </List>

        <Divider sx={{ my: 2 }} />

        <Box sx={{ p: 2 }}>
          <Typography
            variant="caption"
            color="textSecondary"
            sx={{ display: 'block', mb: 1 }}
          >
            Powered by
          </Typography>
          <Typography
            variant="body2"
            color="primary"
            sx={{ fontWeight: 'bold' }}
          >
            OLLI AI
          </Typography>
        </Box>
      </Box>
    </Drawer>
  );
}

export default Sidebar;
