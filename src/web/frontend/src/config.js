const config = {
  apiUrl: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  staticUrl: process.env.REACT_APP_STATIC_URL || 'http://localhost:3000',
  auth: {
    clientId: process.env.REACT_APP_AUTH_CLIENT_ID,
    authority: `https://login.microsoftonline.com/${process.env.REACT_APP_TENANT_ID}`,
    redirectUri: `${process.env.REACT_APP_STATIC_URL}/auth`,
  },
};

export default config;
