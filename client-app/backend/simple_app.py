"""
Simplified Production ML Platform - Demo Backend
FastAPI application for testing without Triton dependency issues
"""

import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import io

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Production ML Platform API (Demo)",
    description="Simplified demo API showcasing the ML platform without Triton dependency",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
TRITON_URL = "http://localhost:8000"
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FORMATS = {"JPEG", "PNG", "JPG"}

def check_triton_health() -> Dict[str, str]:
    """Check if Triton server is accessible."""
    try:
        response = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=2)
        if response.status_code == 200:
            return {"triton_server": "healthy", "model_status": "ready"}
        else:
            return {"triton_server": "not_ready", "model_status": "unknown"}
    except requests.exceptions.RequestException:
        return {"triton_server": "not_accessible", "model_status": "unknown"}

def simulate_ml_inference(image: Image.Image) -> Dict[str, Any]:
    """
    Simulate ML inference results for demo purposes.
    In production, this would call Triton Inference Server.
    """
    
    # Simulate processing time
    time.sleep(0.1)  # 100ms processing time
    
    # Get image properties for simulation
    width, height = image.size
    mode = image.mode
    
    # Simulate different decisions based on image characteristics
    pixel_variance = np.var(np.array(image))
    image_hash = hash(str(np.array(image).tobytes())) % 100
    
    # Create more diverse scenarios based on image properties
    if pixel_variance < 500:
        decision = "REJECTED"
        confidence = 0.89
        category = "low_quality_image"
        safety = "Risk: HIGH, Safe: False"
        safety_reason = "Image quality too low for reliable analysis"
    elif pixel_variance < 1000:
        decision = "REVIEW_REQUIRED"
        confidence = 0.65
        category = "unclear_content"
        safety = "Risk: MEDIUM, Safe: Unknown"
        safety_reason = "Requires human review"
    elif image_hash < 10:  # 10% chance of flagged content
        decision = "FLAGGED"
        confidence = 0.94
        category = "potentially_inappropriate"
        safety = "Risk: HIGH, Safe: False"
        safety_reason = "Content flagged by safety classifier"
    elif image_hash < 20:  # 10% chance of text-heavy content
        decision = "APPROVED"
        confidence = 0.85
        category = "text_heavy_image"
        safety = "Risk: LOW, Safe: True"
        safety_reason = "Text content detected and approved"
        extracted_texts = ["DEMO TEXT", "Sample content", "Image contains readable text"]
    elif pixel_variance > 4000:
        decision = "APPROVED"  
        confidence = 0.92
        category = "high_detail_image"
        safety = "Risk: LOW, Safe: True"
        safety_reason = "High-quality detailed image"
    else:
        decision = "APPROVED"
        confidence = 0.78
        category = "standard_image"
        safety = "Risk: LOW, Safe: True"
        safety_reason = "Standard content approved"
    
    # Simulate extracted text based on category
    if 'extracted_texts' not in locals():
        extracted_texts = ["Sample text"] if np.random.random() > 0.8 else []
    
    # Create metadata based on decision
    safety_result = "REJECT" if decision in ["REJECTED", "FLAGGED"] else "REVIEW" if decision == "REVIEW_REQUIRED" else "APPROVE"
    content_result = "REJECT" if decision == "REJECTED" else "REVIEW" if decision in ["REVIEW_REQUIRED", "FLAGGED"] else "APPROVE"
    text_result = "APPROVE" if not extracted_texts or decision != "FLAGGED" else "REVIEW"
    
    metadata = {
        "decision_factors": {
            "safety_check": {"result": safety_result, "confidence": confidence, "reason": safety_reason},
            "content_check": {"result": content_result, "confidence": confidence, "reason": f"Category: {category}"},  
            "text_check": {"result": text_result, "confidence": 0.95, "reason": f"Text analysis: {len(extracted_texts)} items found"}
        },
        "model_version": "demo_v2.1.0",
        "image_properties": {
            "width": width,
            "height": height, 
            "mode": mode,
            "pixel_variance": float(pixel_variance)
        },
        "processing_pipeline": ["content-classifier", "safety-detector", "ocr-engine", "decision-logic"]
    }
    
    return {
        "moderation_decision": decision,
        "confidence_score": confidence,
        "content_category": category,
        "safety_assessment": safety,
        "extracted_text": extracted_texts,
        "processing_metadata": json.dumps(metadata)
    }

# ===================================
# API ENDPOINTS
# ===================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "Production ML Platform API (Demo)",
        "version": "2.1.0",
        "status": "healthy",
        "mode": "demo_simulation",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    triton_status = check_triton_health()
    
    health_status = {
        "api": "healthy",
        "triton_server": triton_status["triton_server"],
        "model_status": triton_status["model_status"],
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return health_status

@app.get("/models")
async def list_models():
    """List available models (demo simulation)."""
    models = {
        "content-classifier": {"status": "simulated", "type": "image_classification"},
        "safety-detector": {"status": "simulated", "type": "content_safety"},
        "ocr-engine": {"status": "simulated", "type": "text_extraction"},
        "content-moderation-ensemble": {"status": "simulated", "type": "ensemble_pipeline"}
    }
    
    return {"models": models, "demo_mode": True}

@app.post("/moderate-image")
async def moderate_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Image file to moderate")
):
    """
    Main endpoint for content moderation (demo simulation).
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
        
        # Simulate ML inference
        result = simulate_ml_inference(image)
        
        # Add timing information
        processing_time = time.time() - start_time
        result["processing_time_ms"] = round(processing_time * 1000, 2)
        result["timestamp"] = datetime.utcnow().isoformat()
        result["demo_mode"] = True
        
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
    """Batch image moderation (demo simulation)."""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_time = time.time()
    results = []
    
    try:
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                results.append({"filename": file.filename, "error": "Invalid file type"})
                continue
                
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            
            # Simulate inference for this image
            image_result = simulate_ml_inference(image)
            image_result["filename"] = file.filename
            results.append(image_result)
        
        processing_time = time.time() - start_time
        
        return {
            "batch_results": results,
            "total_images": len(files),
            "processed_images": len([r for r in results if "error" not in r]),
            "processing_time_ms": round(processing_time * 1000, 2),
            "timestamp": datetime.utcnow().isoformat(),
            "demo_mode": True
        }
        
    except Exception as e:
        logger.error(f"Error in batch moderation: {e}")
        raise HTTPException(status_code=500, detail="Batch moderation failed")

@app.get("/metrics")
async def get_metrics():
    """Get application metrics (demo simulation)."""
    
    # Simulate some realistic metrics
    return {
        "model_stats": {
            "inference_count": 1247,
            "execution_count": 1247,
            "inference_stats": {
                "success_count": 1239,
                "fail_count": 8,
                "avg_queue_time_ms": 12.5,
                "avg_compute_time_ms": 89.3,
            }
        },
        "demo_mode": True,
        "timestamp": datetime.utcnow().isoformat()
    }

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
        "service": "content-moderation-demo"
    }
    
    logger.info(f"Moderation request logged: {json.dumps(log_entry)}")

# ===================================
# APPLICATION STARTUP
# ===================================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info("Starting Production ML Platform API (Demo Mode)...")
    
    # Check if Triton is available
    triton_status = check_triton_health()
    if triton_status["triton_server"] == "healthy":
        logger.info("Triton server detected and healthy!")
    else:
        logger.info("Running in demo mode - Triton server not accessible")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down Production ML Platform API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "simple_app:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
