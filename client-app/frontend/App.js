/**
 * Production ML Platform - Frontend Demo
 * React application demonstrating Triton-powered content moderation
 */

import React, { useState, useCallback, useEffect } from 'react';
import './App.css';

// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8080';

// Components
const FileUpload = ({ onFileSelect, isLoading }) => {
  const [dragOver, setDragOver] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const imageFiles = files.filter(file => file.type.startsWith('image/'));
    
    if (imageFiles.length > 0) {
      onFileSelect(imageFiles);
    }
  }, [onFileSelect]);

  const handleFileInput = useCallback((e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      onFileSelect(files);
    }
  }, [onFileSelect]);

  return (
    <div 
      className={`file-upload ${dragOver ? 'drag-over' : ''} ${isLoading ? 'loading' : ''}`}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="upload-content">
        <div className="upload-icon">📸</div>
        <h3>Upload Images for Content Moderation</h3>
        <p>Drag & drop images here, or click to browse</p>
        <input
          type="file"
          multiple
          accept="image/*"
          onChange={handleFileInput}
          disabled={isLoading}
          style={{ display: 'none' }}
          id="file-input"
        />
        <label htmlFor="file-input" className="browse-button">
          {isLoading ? 'Processing...' : 'Browse Files'}
        </label>
        <div className="upload-info">
          <small>Supported formats: JPEG, PNG • Max size: 5MB each • Max 10 files</small>
        </div>
      </div>
    </div>
  );
};

const ModerationResult = ({ result, filename }) => {
  const getDecisionColor = (decision) => {
    switch (decision) {
      case 'APPROVED': return '#4CAF50';
      case 'REJECTED': return '#F44336';
      case 'REVIEW_REQUIRED': return '#FF9800';
      default: return '#757575';
    }
  };

  const getDecisionIcon = (decision) => {
    switch (decision) {
      case 'APPROVED': return '✅';
      case 'REJECTED': return '❌';
      case 'REVIEW_REQUIRED': return '⚠️';
      default: return '❓';
    }
  };

  return (
    <div className="moderation-result">
      <div className="result-header">
        <h4>{filename}</h4>
        <div 
          className="decision-badge"
          style={{ backgroundColor: getDecisionColor(result.moderation_decision) }}
        >
          {getDecisionIcon(result.moderation_decision)} {result.moderation_decision}
        </div>
      </div>
      
      <div className="result-details">
        <div className="detail-row">
          <span className="label">Confidence:</span>
          <span className="value">{(result.confidence_score * 100).toFixed(1)}%</span>
        </div>
        
        <div className="detail-row">
          <span className="label">Content Category:</span>
          <span className="value">{result.content_category}</span>
        </div>
        
        <div className="detail-row">
          <span className="label">Safety Assessment:</span>
          <span className="value">{result.safety_assessment}</span>
        </div>
        
        {result.extracted_text && result.extracted_text.length > 0 && (
          <div className="detail-row">
            <span className="label">Extracted Text:</span>
            <span className="value">{result.extracted_text.join(', ')}</span>
          </div>
        )}
        
        <div className="detail-row">
          <span className="label">Processing Time:</span>
          <span className="value">{result.processing_time_ms}ms</span>
        </div>
      </div>
      
      {result.processing_metadata && (
        <details className="metadata-details">
          <summary>Technical Details</summary>
          <pre>{JSON.stringify(JSON.parse(result.processing_metadata), null, 2)}</pre>
        </details>
      )}
    </div>
  );
};

const SystemHealth = ({ health }) => {
  const getStatusColor = (status) => {
    if (status === 'healthy' || status === 'ready') return '#4CAF50';
    if (status === 'not_ready') return '#FF9800';
    if (status.includes('error')) return '#F44336';
    return '#757575';
  };

  const getStatusIcon = (status) => {
    if (status === 'healthy' || status === 'ready') return '🟢';
    if (status === 'not_ready') return '🟡';
    if (status.includes('error')) return '🔴';
    return '⚪';
  };

  return (
    <div className="system-health">
      <h3>System Status</h3>
      <div className="health-indicators">
        <div className="health-item">
          <span className="health-label">API Service:</span>
          <span 
            className="health-status"
            style={{ color: getStatusColor(health.api) }}
          >
            {getStatusIcon(health.api)} {health.api}
          </span>
        </div>
        
        <div className="health-item">
          <span className="health-label">Triton Server:</span>
          <span 
            className="health-status"
            style={{ color: getStatusColor(health.triton_server) }}
          >
            {getStatusIcon(health.triton_server)} {health.triton_server}
          </span>
        </div>
        
        <div className="health-item">
          <span className="health-label">ML Models:</span>
          <span 
            className="health-status"
            style={{ color: getStatusColor(health.model_status) }}
          >
            {getStatusIcon(health.model_status)} {health.model_status}
          </span>
        </div>
      </div>
    </div>
  );
};

// Main App Component
function App() {
  const [results, setResults] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({});
  const [metrics, setMetrics] = useState({});

  // Fetch system health on component mount
  useEffect(() => {
    fetchHealth();
    fetchMetrics();
    
    // Set up polling for health and metrics
    const healthInterval = setInterval(fetchHealth, 30000); // Every 30 seconds
    const metricsInterval = setInterval(fetchMetrics, 10000); // Every 10 seconds
    
    return () => {
      clearInterval(healthInterval);
      clearInterval(metricsInterval);
    };
  }, []);

  const fetchHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setHealth(data);
    } catch (err) {
      console.error('Failed to fetch health:', err);
      setHealth({ api: 'error', triton_server: 'error', model_status: 'error' });
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
    }
  };

  const handleFileSelect = async (files) => {
    setIsLoading(true);
    setError(null);
    
    try {
      if (files.length === 1) {
        // Single file processing
        await processSingleFile(files[0]);
      } else {
        // Batch processing
        await processBatchFiles(files);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  const processSingleFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/moderate-image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process image');
    }

    const result = await response.json();
    setResults(prev => [{
      filename: file.name,
      result: result,
      id: Date.now()
    }, ...prev]);
  };

  const processBatchFiles = async (files) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/batch-moderate`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process batch');
    }

    const data = await response.json();
    const batchResults = data.batch_results.map((result, index) => ({
      filename: result.filename,
      result: result.error ? { error: result.error } : result,
      id: Date.now() + index
    }));

    setResults(prev => [...batchResults, ...prev]);
  };

  const clearResults = () => {
    setResults([]);
    setError(null);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🛡️ Production ML Platform</h1>
        <p>AI-Powered Content Moderation with Triton Inference Server</p>
      </header>

      <div className="app-content">
        <div className="main-section">
          <FileUpload onFileSelect={handleFileSelect} isLoading={isLoading} />
          
          {error && (
            <div className="error-message">
              <h3>❌ Error</h3>
              <p>{error}</p>
            </div>
          )}

          {results.length > 0 && (
            <div className="results-section">
              <div className="results-header">
                <h2>Moderation Results</h2>
                <button onClick={clearResults} className="clear-button">
                  Clear Results
                </button>
              </div>
              
              <div className="results-grid">
                {results.map(({ filename, result, id }) => (
                  <ModerationResult 
                    key={id} 
                    filename={filename} 
                    result={result} 
                  />
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="sidebar">
          <SystemHealth health={health} />
          
          {metrics.model_stats && (
            <div className="metrics-panel">
              <h3>Performance Metrics</h3>
              <div className="metric-item">
                <span className="metric-label">Total Inferences:</span>
                <span className="metric-value">
                  {metrics.model_stats.inference_count.toLocaleString()}
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Success Rate:</span>
                <span className="metric-value">
                  {((metrics.model_stats.inference_stats.success_count / 
                     (metrics.model_stats.inference_stats.success_count + 
                      metrics.model_stats.inference_stats.fail_count)) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="metric-item">
                <span className="metric-label">Avg Queue Time:</span>
                <span className="metric-value">
                  {(metrics.model_stats.inference_stats.queue_time_ns / 1000000).toFixed(1)}ms
                </span>
              </div>
            </div>
          )}
          
          <div className="info-panel">
            <h3>About This Demo</h3>
            <p>
              This application demonstrates a production-ready ML inference pipeline 
              powered by NVIDIA Triton Inference Server.
            </p>
            <ul>
              <li>🔍 Content Classification</li>
              <li>🛡️ Safety Detection</li>
              <li>📝 Text Extraction (OCR)</li>
              <li>🤖 Ensemble Decision Logic</li>
            </ul>
            <p>
              Built with enterprise patterns: auto-scaling, monitoring, 
              CI/CD, and production security.
            </p>
          </div>
        </div>
      </div>

      <footer className="app-footer">
        <p>
          Production ML Platform v2.1.0 | 
          Powered by <strong>NVIDIA Triton Inference Server</strong>
        </p>
      </footer>
    </div>
  );
}

export default App;
