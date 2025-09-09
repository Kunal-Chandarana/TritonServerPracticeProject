"""
Production ML Platform - Demo Client Backend
FastAPI application demonstrating Triton Inference Server integration
"""

import asyncio
import io
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import numpy as np
import tritonclient.http as httpclient
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production ML Platform API",
    description="Demo API for Triton-powered content moderation system",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TRITON_URL = "localhost:8000"  # In production: service discovery
MODEL_NAME = "content-moderation-ensemble"
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FORMATS = {"JPEG", "PNG", "JPG"}

class TritonClient:
    """Production Triton client with connection pooling and error handling."""
    
    def __init__(self, url: str):
        self.url = url
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Triton server."""
        try:
            self.client = httpclient.InferenceServerClient(
                url=self.url,
                verbose=False,
                connection_timeout=30.0,
                network_timeout=60.0
            )
            
            # Verify server health
            if not self.client.is_server_ready():
                raise ConnectionError("Triton server is not ready")
                
            logger.info(f"Connected to Triton server at {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Triton server: {e}")
            raise
    
    def is_model_ready(self, model_name: str) -> bool:
        """Check if specific model is ready."""
        try:
            return self.client.is_model_ready(model_name)
        except Exception as e:
            logger.error(f"Error checking model readiness: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Dict:
        """Get model metadata."""
        try:
            metadata = self.client.get_model_metadata(model_name)
            return {
                "name": metadata.name,
                "platform": metadata.platform,
                "versions": metadata.versions,
                "inputs": [{"name": inp.name, "datatype": inp.datatype, "shape": inp.shape} 
                          for inp in metadata.inputs],
                "outputs": [{"name": out.name, "datatype": out.datatype, "shape": out.shape} 
                           for out in metadata.outputs]
            }
        except Exception as e:
            logger.error(f"Error getting model metadata: {e}")
            return {}

# Initialize Triton client
triton_client = TritonClient(TRITON_URL)

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model inference.
    Convert to RGB, resize, and normalize.
    """
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Ensure uint8 format
        if image_array.dtype != np.uint8:
            image_array = image_array.astype(np.uint8)
        
        # Add batch dimension and return
        return np.expand_dims(image_array, axis=0)
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise HTTPException(status_code=400, detail="Image preprocessing failed")

def create_inference_request(image_data: np.ndarray) -> List[httpclient.InferInput]:
    """Create Triton inference request inputs."""
    try:
        inputs = []
        
        # Create input tensor
        input_tensor = httpclient.InferInput(
            "input_image", 
            image_data.shape, 
            "UINT8"
        )
        input_tensor.set_data_from_numpy(image_data)
        inputs.append(input_tensor)
        
        return inputs
        
    except Exception as e:
        logger.error(f"Error creating inference request: {e}")
        raise HTTPException(status_code=500, detail="Request creation failed")

def parse_inference_response(response) -> Dict[str, Any]:
    """Parse Triton inference response."""
    try:
        result = {}
        
        # Extract all outputs
        result["moderation_decision"] = response.as_numpy("moderation_decision")[0].decode('utf-8')
        result["confidence_score"] = float(response.as_numpy("confidence_score")[0])
        result["content_category"] = response.as_numpy("content_category")[0].decode('utf-8')
        result["safety_assessment"] = response.as_numpy("safety_assessment")[0].decode('utf-8')
        
        # Extract text (may be multiple entries)
        extracted_text = response.as_numpy("extracted_text")
        result["extracted_text"] = [text.decode('utf-8') for text in extracted_text]
        
        # Parse metadata
        metadata_json = response.as_numpy("processing_metadata")[0].decode('utf-8')
        result["processing_metadata"] = json.loads(metadata_json)
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing inference response: {e}")
        raise HTTPException(status_code=500, detail="Response parsing failed")

# ===================================
# API ENDPOINTS
# ===================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Production ML Platform API",
        "version": "2.1.0",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "api": "healthy",
        "triton_server": "unknown",
        "model_status": "unknown",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    try:
        # Check Triton server
        if triton_client.client.is_server_ready():
            health_status["triton_server"] = "healthy"
            
            # Check model
            if triton_client.is_model_ready(MODEL_NAME):
                health_status["model_status"] = "ready"
            else:
                health_status["model_status"] = "not_ready"
        else:
            health_status["triton_server"] = "not_ready"
            
    except Exception as e:
        health_status["triton_server"] = f"error: {str(e)}"
        
    return health_status

@app.get("/models")
async def list_models():
    """List available models."""
    try:
        # In production, this would query Triton's model repository
        models = ["content-classifier", "safety-detector", "ocr-engine", "content-moderation-ensemble"]
        
        model_info = {}
        for model in models:
            if triton_client.is_model_ready(model):
                model_info[model] = {
                    "status": "ready",
                    "metadata": triton_client.get_model_metadata(model)
                }
            else:
                model_info[model] = {"status": "not_ready"}
                
        return {"models": model_info}
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail="Failed to list models")

@app.post("/moderate-image")
async def moderate_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to moderate")
):
    """
    Main endpoint for content moderation.
    Processes uploaded image through the complete ML pipeline.
    """
    start_time = time.time()
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size and file.size > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="Image too large (max 5MB)")
    
    try:
        # Read and validate image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.format not in ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail=f"Unsupported format. Use: {ALLOWED_FORMATS}")
        
        # Preprocess image
        image_data = preprocess_image(image)
        
        # Create inference request
        inputs = create_inference_request(image_data)
        
        # Define expected outputs
        outputs = [
            httpclient.InferRequestedOutput("moderation_decision"),
            httpclient.InferRequestedOutput("confidence_score"),
            httpclient.InferRequestedOutput("content_category"),
            httpclient.InferRequestedOutput("safety_assessment"),
            httpclient.InferRequestedOutput("extracted_text"),
            httpclient.InferRequestedOutput("processing_metadata"),
        ]
        
        # Make inference request
        response = triton_client.client.infer(
            model_name=MODEL_NAME,
            inputs=inputs,
            outputs=outputs
        )
        
        # Parse response
        result = parse_inference_response(response)
        
        # Add timing information
        processing_time = time.time() - start_time
        result["processing_time_ms"] = round(processing_time * 1000, 2)
        result["timestamp"] = datetime.utcnow().isoformat()
        
        # Log request for monitoring (background task)
        background_tasks.add_task(
            log_moderation_request,
            file.filename,
            result["moderation_decision"],
            processing_time
        )
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in image moderation: {e}")
        raise HTTPException(status_code=500, detail="Image moderation failed")

@app.post("/batch-moderate")
async def batch_moderate_images(
    files: List[UploadFile] = File(..., description="Multiple image files to moderate")
):
    """
    Batch image moderation endpoint.
    Processes multiple images efficiently using Triton's batching capabilities.
    """
    if len(files) > 10:  # Production limit
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_time = time.time()
    results = []
    
    try:
        # Process all images
        batch_data = []
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({"filename": file.filename, "error": "Invalid file type"})
                continue
                
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            image_data = preprocess_image(image)
            batch_data.append((file.filename, image_data))
        
        # Create batch inference request
        if batch_data:
            # Combine all images into single batch
            combined_images = np.concatenate([data[1] for _, data in batch_data], axis=0)
            
            inputs = create_inference_request(combined_images)
            outputs = [
                httpclient.InferRequestedOutput("moderation_decision"),
                httpclient.InferRequestedOutput("confidence_score"),
                httpclient.InferRequestedOutput("content_category"),
                httpclient.InferRequestedOutput("safety_assessment"),
                httpclient.InferRequestedOutput("extracted_text"),
                httpclient.InferRequestedOutput("processing_metadata"),
            ]
            
            # Make batch inference
            response = triton_client.client.infer(
                model_name=MODEL_NAME,
                inputs=inputs,
                outputs=outputs
            )
            
            # Parse batch response
            for i, (filename, _) in enumerate(batch_data):
                try:
                    # Extract results for this image from batch response
                    image_result = {
                        "filename": filename,
                        "moderation_decision": response.as_numpy("moderation_decision")[i].decode('utf-8'),
                        "confidence_score": float(response.as_numpy("confidence_score")[i]),
                        "content_category": response.as_numpy("content_category")[i].decode('utf-8'),
                        "safety_assessment": response.as_numpy("safety_assessment")[i].decode('utf-8'),
                        "extracted_text": [text.decode('utf-8') for text in response.as_numpy("extracted_text")],
                    }
                    results.append(image_result)
                except Exception as e:
                    results.append({"filename": filename, "error": str(e)})
        
        processing_time = time.time() - start_time
        
        return {
            "batch_results": results,
            "total_images": len(files),
            "processed_images": len([r for r in results if "error" not in r]),
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in batch moderation: {e}")
        raise HTTPException(status_code=500, detail="Batch moderation failed")

@app.get("/metrics")
async def get_metrics():
    """
    Get application and model metrics.
    In production, this would integrate with Prometheus.
    """
    try:
        # Get Triton server statistics
        stats = triton_client.client.get_inference_statistics(MODEL_NAME)
        
        return {
            "model_stats": {
                "inference_count": stats.model_stats[0].inference_count,
                "execution_count": stats.model_stats[0].execution_count,
                "inference_stats": {
                    "success_count": stats.model_stats[0].inference_stats.success.count,
                    "fail_count": stats.model_stats[0].inference_stats.fail.count,
                    "queue_time_ns": stats.model_stats[0].inference_stats.queue.total_time_ns,
                    "compute_time_ns": stats.model_stats[0].inference_stats.compute_input.total_time_ns,
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return {"error": "Failed to get metrics", "timestamp": datetime.utcnow().isoformat()}

# ===================================
# BACKGROUND TASKS
# ===================================

async def log_moderation_request(filename: str, decision: str, processing_time: float):
    """Log moderation request for monitoring and analytics."""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "filename": filename,
        "decision": decision,
        "processing_time": processing_time,
        "service": "content-moderation"
    }
    
    # In production: send to logging service, metrics collector, etc.
    logger.info(f"Moderation request logged: {json.dumps(log_entry)}")

# ===================================
# APPLICATION STARTUP
# ===================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting Production ML Platform API...")
    
    # Verify Triton connection
    try:
        if not triton_client.client.is_server_ready():
            logger.error("Triton server is not ready!")
        else:
            logger.info("Triton server connection verified")
            
        # Check if main model is ready
        if triton_client.is_model_ready(MODEL_NAME):
            logger.info(f"Model '{MODEL_NAME}' is ready")
        else:
            logger.warning(f"Model '{MODEL_NAME}' is not ready")
            
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Production ML Platform API...")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
