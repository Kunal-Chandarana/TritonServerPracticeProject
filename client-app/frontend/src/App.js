import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Alert,
  CircularProgress,
  Grid,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
} from '@mui/material';
import {
  CloudUpload,
  CheckCircle,
  Warning,
  Error,
  Info,
  Speed,
  Security,
  TextFields,
  Image as ImageIcon,
} from '@mui/icons-material';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import axios from 'axios';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#667eea',
    },
    secondary: {
      main: '#764ba2',
    },
    success: {
      main: '#4caf50',
    },
    warning: {
      main: '#ff9800',
    },
    error: {
      main: '#f44336',
    },
  },
});

const API_BASE_URL = 'http://localhost:8080';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [apiHealth, setApiHealth] = useState(null);

  // Check API health on component mount
  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`);
      setApiHealth(response.data);
    } catch (err) {
      console.error('API health check failed:', err);
      setApiHealth({ api: 'unhealthy', error: err.message });
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const moderateImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post(`${API_BASE_URL}/moderate-image`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to moderate image');
    } finally {
      setLoading(false);
    }
  };

  const getDecisionColor = (decision) => {
    switch (decision) {
      case 'APPROVED': return 'success';
      case 'REJECTED': return 'error';
      case 'FLAGGED': return 'error';
      case 'REVIEW_REQUIRED': return 'warning';
      default: return 'default';
    }
  };

  const getDecisionIcon = (decision) => {
    switch (decision) {
      case 'APPROVED': return <CheckCircle />;
      case 'REJECTED': return <Error />;
      case 'FLAGGED': return <Warning />;
      case 'REVIEW_REQUIRED': return <Info />;
      default: return <Info />;
    }
  };

  const parseMetadata = (metadataString) => {
    try {
      return JSON.parse(metadataString);
    } catch {
      return null;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Header */}
        <Paper elevation={3} sx={{ p: 4, mb: 4, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            🏭 Production ML Platform
          </Typography>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Content Moderation Demo
          </Typography>
          {apiHealth && (
            <Box sx={{ mt: 2 }}>
              <Chip
                icon={apiHealth.api === 'healthy' ? <CheckCircle /> : <Error />}
                label={`API: ${apiHealth.api} | Models: ${apiHealth.model_status || 'unknown'}`}
                color={apiHealth.api === 'healthy' ? 'success' : 'error'}
                variant="outlined"
              />
              {apiHealth.demo_mode && (
                <Chip
                  label="Demo Mode"
                  color="info"
                  variant="outlined"
                  sx={{ ml: 1 }}
                />
              )}
            </Box>
          )}
        </Paper>

        <Grid container spacing={4}>
          {/* Upload Section */}
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom>
                📤 Upload Image
              </Typography>
              
              {/* File Drop Zone */}
              <Box
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                sx={{
                  border: `2px dashed ${dragActive ? theme.palette.primary.main : '#ccc'}`,
                  borderRadius: 2,
                  p: 4,
                  textAlign: 'center',
                  backgroundColor: dragActive ? 'action.hover' : 'background.paper',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  mb: 2,
                }}
                onClick={() => document.getElementById('file-input').click()}
              >
                <CloudUpload sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                <Typography variant="h6" gutterBottom>
                  {selectedFile ? selectedFile.name : 'Drop image here or click to select'}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Supports JPEG, PNG, JPG (max 5MB)
                </Typography>
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  style={{ display: 'none' }}
                />
              </Box>

              {/* Action Button */}
              <Button
                variant="contained"
                fullWidth
                size="large"
                onClick={moderateImage}
                disabled={!selectedFile || loading}
                startIcon={loading ? <CircularProgress size={20} /> : <Security />}
              >
                {loading ? 'Analyzing...' : 'Moderate Image'}
              </Button>

              {error && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {error}
                </Alert>
              )}
            </Paper>
          </Grid>

          {/* Results Section */}
          <Grid item xs={12} md={6}>
            <Paper elevation={3} sx={{ p: 4 }}>
              <Typography variant="h5" gutterBottom>
                📊 Analysis Results
              </Typography>
              
              {result ? (
                <Box>
                  {/* Main Decision */}
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        {getDecisionIcon(result.moderation_decision)}
                        <Typography variant="h6" sx={{ ml: 1 }}>
                          Decision
                        </Typography>
                      </Box>
                      <Chip
                        label={result.moderation_decision}
                        color={getDecisionColor(result.moderation_decision)}
                        size="large"
                        sx={{ mb: 1 }}
                      />
                      <Typography variant="body2" color="text.secondary">
                        Confidence: {(result.confidence_score * 100).toFixed(1)}%
                      </Typography>
                    </CardContent>
                  </Card>

                  {/* Details */}
                  <Card sx={{ mb: 2 }}>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        📋 Details
                      </Typography>
                      <List dense>
                        <ListItem>
                          <ListItemIcon><ImageIcon /></ListItemIcon>
                          <ListItemText
                            primary="Content Category"
                            secondary={result.content_category}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><Security /></ListItemIcon>
                          <ListItemText
                            primary="Safety Assessment"
                            secondary={result.safety_assessment}
                          />
                        </ListItem>
                        <ListItem>
                          <ListItemIcon><Speed /></ListItemIcon>
                          <ListItemText
                            primary="Processing Time"
                            secondary={`${result.processing_time_ms.toFixed(1)}ms`}
                          />
                        </ListItem>
                        {result.extracted_text && result.extracted_text.length > 0 && (
                          <ListItem>
                            <ListItemIcon><TextFields /></ListItemIcon>
                            <ListItemText
                              primary="Extracted Text"
                              secondary={result.extracted_text.join(', ')}
                            />
                          </ListItem>
                        )}
                      </List>
                    </CardContent>
                  </Card>

                  {/* Technical Details */}
                  {parseMetadata(result.processing_metadata) && (
                    <Card>
                      <CardContent>
                        <Typography variant="h6" gutterBottom>
                          🔧 Technical Details
                        </Typography>
                        {(() => {
                          const metadata = parseMetadata(result.processing_metadata);
                          return (
                            <Box>
                              <Typography variant="subtitle2" gutterBottom>
                                Decision Factors:
                              </Typography>
                              {Object.entries(metadata.decision_factors || {}).map(([key, value]) => (
                                <Chip
                                  key={key}
                                  label={`${key}: ${value.result}`}
                                  color={value.result === 'APPROVE' ? 'success' : value.result === 'REJECT' ? 'error' : 'warning'}
                                  variant="outlined"
                                  size="small"
                                  sx={{ mr: 1, mb: 1 }}
                                />
                              ))}
                              <Divider sx={{ my: 2 }} />
                              <Typography variant="body2" color="text.secondary">
                                Model Version: {metadata.model_version}
                              </Typography>
                              <Typography variant="body2" color="text.secondary">
                                Image: {metadata.image_properties?.width}x{metadata.image_properties?.height}
                              </Typography>
                            </Box>
                          );
                        })()}
                      </CardContent>
                    </Card>
                  )}
                </Box>
              ) : (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="body1" color="text.secondary">
                    Upload an image to see moderation results
                  </Typography>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>

        {/* Footer */}
        <Paper elevation={1} sx={{ p: 2, mt: 4, textAlign: 'center' }}>
          <Typography variant="body2" color="text.secondary">
            Production ML Platform Demo • Powered by Triton Inference Server
          </Typography>
        </Paper>
      </Container>
    </ThemeProvider>
  );
}

export default App;
