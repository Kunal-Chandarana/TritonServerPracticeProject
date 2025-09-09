#!/bin/bash
# Local Development Startup Script for Production ML Platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
TRITON_CONTAINER_NAME="triton-local"
TRITON_IMAGE="nvcr.io/nvidia/tritonserver:25.08-py3"
MODEL_REPO_PATH="$(pwd)/models"

# Helper functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker Desktop."
    fi
    
    # Check if Docker is running
    if ! docker info &> /dev/null; then
        error "Docker is not running. Please start Docker Desktop."
    fi
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is not installed."
    fi
    
    success "Prerequisites check passed"
}

# Start Triton Inference Server
start_triton() {
    log "Starting Triton Inference Server..."
    
    # Stop existing container if running
    if docker ps -q -f name=$TRITON_CONTAINER_NAME | grep -q .; then
        warning "Stopping existing Triton container..."
        docker stop $TRITON_CONTAINER_NAME
    fi
    
    # Remove existing container if exists
    if docker ps -aq -f name=$TRITON_CONTAINER_NAME | grep -q .; then
        docker rm $TRITON_CONTAINER_NAME
    fi
    
    # Pull latest image
    log "Pulling Triton image (this may take a few minutes)..."
    docker pull $TRITON_IMAGE
    
    # Start Triton container
    log "Starting Triton container..."
    docker run -d --name $TRITON_CONTAINER_NAME \
        -p 8000:8000 -p 8001:8001 -p 8002:8002 \
        -v "$MODEL_REPO_PATH:/models" \
        $TRITON_IMAGE \
        tritonserver \
        --model-repository=/models \
        --model-control-mode=explicit \
        --log-verbose=1 \
        --log-info=true \
        --allow-metrics=true \
        --exit-timeout-secs=30
    
    # Wait for Triton to be ready
    log "Waiting for Triton to be ready..."
    local max_attempts=30
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s localhost:8000/v2/health/ready &> /dev/null; then
            success "Triton server is ready!"
            break
        fi
        
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    if [ $attempt -eq $max_attempts ]; then
        error "Triton server failed to start. Check logs with: docker logs $TRITON_CONTAINER_NAME"
    fi
}

# Load models into Triton
load_models() {
    log "Loading models into Triton..."
    
    # Note: In a real setup, you'd load individual models first
    # For this demo, we'll try to load the ensemble directly
    
    models_to_load=(
        "content-classifier"
        "safety-detector" 
        "ocr-engine"
        "moderation-decision-logic"
        "content-moderation-ensemble"
    )
    
    for model in "${models_to_load[@]}"; do
        log "Loading model: $model"
        if curl -s -X POST "localhost:8000/v2/repository/models/$model/load" | grep -q "error"; then
            warning "Failed to load model: $model (this is expected for demo models)"
        else
            success "Model loaded: $model"
        fi
        sleep 1
    done
}

# Setup Python backend environment
setup_backend() {
    log "Setting up backend environment..."
    
    cd client-app/backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install dependencies
    log "Installing Python dependencies..."
    pip install -r requirements.txt
    
    success "Backend environment setup complete"
    cd ../..
}

# Display startup information
show_info() {
    echo ""
    echo "🎉 Production ML Platform is starting up!"
    echo ""
    echo "📊 Service Endpoints:"
    echo "  • Triton HTTP API:  http://localhost:8000"
    echo "  • Triton gRPC API:  localhost:8001" 
    echo "  • Triton Metrics:   http://localhost:8002/metrics"
    echo "  • Backend API:      http://localhost:8080 (start manually)"
    echo "  • API Docs:         http://localhost:8080/docs"
    echo ""
    echo "🔧 Useful Commands:"
    echo "  • Check Triton health: curl localhost:8000/v2/health/ready"
    echo "  • List models:         curl localhost:8000/v2/models"
    echo "  • View Triton logs:    docker logs -f $TRITON_CONTAINER_NAME"
    echo "  • Stop Triton:         docker stop $TRITON_CONTAINER_NAME"
    echo ""
    echo "🚀 To start the backend API:"
    echo "  cd client-app/backend"
    echo "  source venv/bin/activate"
    echo "  uvicorn app:app --host 0.0.0.0 --port 8080 --reload"
    echo ""
}

# Main execution
main() {
    echo "🏭 Production ML Platform - Local Setup"
    echo "========================================"
    
    check_prerequisites
    start_triton
    load_models
    setup_backend
    show_info
    
    success "Local setup complete! 🎉"
    
    # Ask if user wants to start backend automatically
    read -p "Start backend API now? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Starting backend API..."
        cd client-app/backend
        source venv/bin/activate
        uvicorn app:app --host 0.0.0.0 --port 8080 --reload
    else
        echo "Run the backend manually when ready!"
    fi
}

# Handle script interruption
cleanup() {
    echo ""
    warning "Script interrupted. Cleaning up..."
    if docker ps -q -f name=$TRITON_CONTAINER_NAME | grep -q .; then
        docker stop $TRITON_CONTAINER_NAME
    fi
    exit 1
}

trap cleanup INT TERM

# Run main function
main "$@"
