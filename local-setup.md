# 🚀 Local Development Setup Guide

This guide will help you run the Production ML Platform locally for development and testing.

## 📋 Prerequisites

### Required Software
- **Docker Desktop** (with at least 8GB RAM allocated)
- **Python 3.8+** 
- **Node.js 16+** and npm
- **Git**

### Optional (for Kubernetes testing)
- **kubectl**
- **Minikube** or **Docker Desktop Kubernetes**

## 🎯 Quick Start (5 minutes)

### Step 1: Start Triton Server

```bash
# Navigate to project directory
cd /Users/kchandarana/server/production-ml-platform

# Start Triton server with our models
docker run --rm -d --name triton-local \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/models:/models \
  nvcr.io/nvidia/tritonserver:25.08-py3 \
  tritonserver --model-repository=/models --model-control-mode=explicit --log-verbose=1

# Wait for server to start (about 30 seconds)
echo "⏳ Waiting for Triton to start..."
sleep 30

# Check server health
curl localhost:8000/v2/health/ready && echo "✅ Triton is ready!"
```

### Step 2: Load Models

```bash
# Load our model ensemble
echo "📥 Loading content moderation ensemble..."
curl -X POST localhost:8000/v2/repository/models/content-moderation-ensemble/load

# Verify model is ready
curl localhost:8000/v2/models/content-moderation-ensemble/ready && echo "✅ Model ready!"
```

### Step 3: Start Backend API

```bash
# Open new terminal and navigate to backend
cd client-app/backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn tritonclient[all] pillow python-multipart

# Start the API server
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

### Step 4: Start Frontend (Optional)

```bash
# Open another terminal and navigate to frontend
cd client-app/frontend

# Install dependencies
npm init -y
npm install react react-dom react-scripts

# Start development server
npm start
```

## 🧪 Test the System

### Test 1: API Health Check
```bash
curl localhost:8080/health
```

### Test 2: Upload Test Image
```bash
# Create a simple test image (or use any image file)
curl -X POST "http://localhost:8080/moderate-image" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

### Test 3: Direct Triton Access
```bash
# Check available models
curl localhost:8000/v2/models

# Get model metadata
curl localhost:8000/v2/models/content-moderation-ensemble
```

## 🔧 Development Workflow

### Making Changes to Models
1. Edit model configurations in `models/*/config.pbtxt`
2. Restart Triton container to pick up changes
3. Reload models via API

### Making Changes to Backend
- API will auto-reload thanks to `--reload` flag
- Just save your changes and test

### Making Changes to Frontend
- React will auto-reload in development mode
- Changes appear immediately in browser

## 🐛 Troubleshooting

### Common Issues

**Triton won't start:**
```bash
# Check Docker logs
docker logs triton-local

# Common fixes:
# 1. Increase Docker memory to 8GB+
# 2. Make sure models directory is properly mounted
# 3. Check for port conflicts
```

**Models won't load:**
```bash
# Check model configuration syntax
# Look for typos in config.pbtxt files
# Verify model files exist (we're using dummy configs)

# Check Triton logs for specific errors
docker logs triton-local | grep ERROR
```

**Backend API errors:**
```bash
# Check if Triton is accessible
curl localhost:8000/v2/health/ready

# Verify Python dependencies
pip list | grep tritonclient

# Check API logs for specific errors
```

**Frontend won't connect:**
- Verify backend is running on port 8080
- Check CORS settings in backend
- Ensure no firewall blocking connections

## 📊 Monitoring Your Local Setup

### View Triton Metrics
```bash
curl localhost:8002/metrics | grep nv_inference
```

### Check System Resources
```bash
# Monitor Docker containers
docker stats triton-local

# Check API performance
curl localhost:8080/metrics
```

## 🧹 Cleanup

### Stop All Services
```bash
# Stop Triton
docker stop triton-local

# Stop backend (Ctrl+C in terminal)
# Stop frontend (Ctrl+C in terminal)
```

### Remove Docker Images (Optional)
```bash
docker rmi nvcr.io/nvidia/tritonserver:25.08-py3
```

## 🎯 Next Steps

Once running locally:

1. **Experiment with Models** - Try different configurations
2. **Test Performance** - Use different batch sizes and concurrency
3. **Add Real Models** - Replace dummy models with actual ONNX files
4. **Try Kubernetes** - Deploy to local Minikube cluster
5. **Explore Monitoring** - Set up Prometheus and Grafana locally

## 💡 Development Tips

- Use `docker logs -f triton-local` to watch Triton logs in real-time
- Backend API docs available at `http://localhost:8080/docs`
- Triton server info at `http://localhost:8000/v2`
- Monitor resource usage with `docker stats`
- Use Postman or curl for API testing

Happy coding! 🚀
