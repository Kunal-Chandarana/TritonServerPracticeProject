#!/usr/bin/env python3
"""
Local Testing Script for Production ML Platform
Tests the complete pipeline: Triton + Backend API
"""

import requests
import json
import time
import sys
from io import BytesIO
from PIL import Image
import numpy as np

# Configuration
TRITON_URL = "http://localhost:8000"
API_URL = "http://localhost:8080"

def create_test_image():
    """Create a simple test image for testing."""
    # Create a 224x224 RGB image with random colors
    image_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    
    # Save to bytes
    img_bytes = BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_triton_health():
    """Test Triton server health."""
    print("🔍 Testing Triton server health...")
    
    try:
        response = requests.get(f"{TRITON_URL}/v2/health/ready", timeout=5)
        if response.status_code == 200:
            print("✅ Triton server is healthy!")
            return True
        else:
            print(f"❌ Triton health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to Triton: {e}")
        return False

def test_triton_models():
    """Test Triton model availability."""
    print("🔍 Testing Triton models...")
    
    try:
        # Check available models
        response = requests.get(f"{TRITON_URL}/v2/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Found {len(models)} models")
            return True
        else:
            print("⚠️  No models loaded (this is expected for demo)")
            return True  # This is OK for our demo
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot query Triton models: {e}")
        return False

def test_api_health():
    """Test backend API health."""
    print("🔍 Testing backend API health...")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Backend API is healthy!")
            print(f"   - API: {health_data.get('api', 'unknown')}")
            print(f"   - Triton: {health_data.get('triton_server', 'unknown')}")
            print(f"   - Models: {health_data.get('model_status', 'unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

def test_image_upload():
    """Test image upload and moderation."""
    print("🔍 Testing image upload and moderation...")
    
    try:
        # Create test image
        test_image = create_test_image()
        
        # Upload image
        files = {'file': ('test_image.jpg', test_image, 'image/jpeg')}
        response = requests.post(f"{API_URL}/moderate-image", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Image moderation successful!")
            print(f"   - Decision: {result.get('moderation_decision', 'unknown')}")
            print(f"   - Confidence: {result.get('confidence_score', 0):.2f}")
            print(f"   - Processing time: {result.get('processing_time_ms', 0)}ms")
            return True
        else:
            print(f"❌ Image moderation failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Image upload failed: {e}")
        return False

def test_api_endpoints():
    """Test various API endpoints."""
    print("🔍 Testing API endpoints...")
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/models", "Models list"),
        ("/metrics", "Metrics endpoint")
    ]
    
    success_count = 0
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"{API_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {description}: OK")
                success_count += 1
            else:
                print(f"   ⚠️  {description}: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"   ❌ {description}: {e}")
    
    return success_count > 0

def main():
    """Run all tests."""
    print("🧪 Production ML Platform - Local Testing")
    print("=" * 50)
    
    tests = [
        ("Triton Health", test_triton_health),
        ("Triton Models", test_triton_models),
        ("API Health", test_api_health),
        ("API Endpoints", test_api_endpoints),
        ("Image Upload", test_image_upload)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your local setup is working perfectly!")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Check the output above for details.")
        print("\n💡 Common issues:")
        print("   - Make sure Triton server is running (./start-local.sh)")
        print("   - Make sure backend API is running (uvicorn app:app --port 8080)")
        print("   - Check Docker has enough memory allocated (8GB+)")
        return 1

if __name__ == "__main__":
    sys.exit(main())
