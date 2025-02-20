import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Avatar,
  Menu,
  MenuItem,
} from '@mui/material';
import { AccountCircle, Notifications } from '@mui/icons-material';


function Header() {
  const [anchorEl, setAnchorEl] = React.useState(null);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  return (
    <AppBar
      position="fixed"
      sx={{
        zIndex: (theme) => theme.zIndex.drawer + 1,
        backgroundColor: 'white',
        color: 'primary.main',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
      }}
    >
      <Toolbar>
        <Typography
          variant="h5"
          sx={{
            fontWeight: 700,
            color: 'primary.main',
            mr: 2,
          }}
        >
          OLLI
        </Typography>
        <Typography
          variant="h6"
          noWrap
          component="div"
          sx={{
            flexGrow: 1,
            color: 'primary.main',
            fontWeight: 600,
          }}
        >
          Sales Forecasting
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center' }}>
          <IconButton
            size="large"
            color="primary"
            sx={{ mr: 2 }}
          >
            <Notifications />
          </IconButton>

          <IconButton
            size="large"
            onClick={handleMenu}
            color="primary"
          >
            <AccountCircle />
          </IconButton>
          <Menu
            id="menu-appbar"
            anchorEl={anchorEl}
            anchorOrigin={{
              vertical: 'bottom',
              horizontal: 'right',
            }}
            keepMounted
            transformOrigin={{
              vertical: 'top',
              horizontal: 'right',
            }}
            open={Boolean(anchorEl)}
            onClose={handleClose}
          >
            <MenuItem onClick={handleClose}>Profile</MenuItem>
            <MenuItem onClick={handleClose}>Settings</MenuItem>
            <MenuItem onClick={handleClose}>Logout</MenuItem>
          </Menu>
        </Box>
      </Toolbar>
    </AppBar>
  );
}

export default Header;
