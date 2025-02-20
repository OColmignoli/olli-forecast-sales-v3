import React, { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Alert,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import { Upload, CheckCircle, Error } from '@mui/icons-material';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';

function DataUpload() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);

  const onDrop = (acceptedFiles) => {
    const file = acceptedFiles[0];
    setFile(file);
    setError(null);
    setSuccess(false);

    // Preview file contents
    const reader = new FileReader();
    reader.onload = () => {
      const csv = reader.result;
      const lines = csv.split('\n');
      const headers = lines[0].split(',');
      const data = lines.slice(1, 6).map(line => line.split(','));
      setPreview({ headers, data });
    };
    reader.readAsText(file);
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
    },
    maxFiles: 1,
  });

  const handleUpload = async () => {
    if (!file) return;

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/api/data/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setSuccess(true);
      setFile(null);
      setPreview(null);
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" sx={{ mb: 4 }}>
        Upload Sales Data
      </Typography>

      <Card sx={{ mb: 4 }}>
        <CardContent>
          <Box
            {...getRootProps()}
            sx={{
              border: '2px dashed',
              borderColor: isDragActive ? 'primary.main' : 'grey.300',
              borderRadius: 2,
              p: 3,
              textAlign: 'center',
              cursor: 'pointer',
              mb: 3,
            }}
          >
            <input {...getInputProps()} />
            <Upload sx={{ fontSize: 48, color: 'primary.main', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              Drag and drop your CSV file here
            </Typography>
            <Typography variant="body2" color="textSecondary">
              or click to select file
            </Typography>
          </Box>

          {file && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Selected file: {file.name}
              </Typography>
              <Button
                variant="contained"
                onClick={handleUpload}
                disabled={uploading}
                sx={{ mr: 2 }}
              >
                Upload Data
              </Button>
              <Button
                variant="outlined"
                onClick={() => {
                  setFile(null);
                  setPreview(null);
                }}
                disabled={uploading}
              >
                Cancel
              </Button>
            </Box>
          )}

          {uploading && (
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
              Data uploaded successfully!
            </Alert>
          )}
        </CardContent>
      </Card>

      {preview && (
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Data Preview
            </Typography>
            <TableContainer component={Paper}>
              <Table size="small">
                <TableHead>
                  <TableRow>
                    {preview.headers.map((header, index) => (
                      <TableCell key={index}>{header}</TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {preview.data.map((row, rowIndex) => (
                    <TableRow key={rowIndex}>
                      {row.map((cell, cellIndex) => (
                        <TableCell key={cellIndex}>{cell}</TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Card>
      )}
    </Box>
  );
}

export default DataUpload;
