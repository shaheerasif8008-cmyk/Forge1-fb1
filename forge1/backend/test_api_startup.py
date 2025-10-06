#!/usr/bin/env python3
"""
Test script to verify API server startup and endpoint registration

This script tests that the FastAPI application can start successfully
with all integrated API routers and that endpoints are properly registered.
"""

import asyncio
import logging
import sys
import time
from typing import List, Dict, Any

import httpx
import uvicorn
from multiprocessing import Process

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def start_server():
    """Start the FastAPI server in a separate process"""
    try:
        # Import here to avoid issues with multiprocessing
        from forge1.main import app
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

async def test_server_startup():
    """Test that the server starts and responds"""
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get("http://127.0.0.1:8000/health", timeout=5.0)
                if response.status_code == 200:
                    logger.info("✅ Server is responding to health checks")
                    return True
        except Exception:
            if attempt < max_attempts - 1:
                logger.info(f"Waiting for server to start... (attempt {attempt + 1}/{max_attempts})")
                await asyncio.sleep(1)
            else:
                logger.error("❌ Server failed to start within timeout")
                return False
    return False

async def test_api_endpoints():
    """Test that all API endpoints are registered"""
    endpoints_to_test = [
        # Core endpoints
        {"path": "/", "method": "GET", "description": "Root endpoint"},
        {"path": "/health", "method": "GET", "description": "Health check"},
        {"path": "/docs", "method": "GET", "description": "API documentation"},
        
        # Employee Lifecycle API
        {"path": "/api/v1/employees/health", "method": "GET", "description": "Employee API health"},
        
        # Other API endpoints (should return 422 or 400 for missing data, not 404)
        {"path": "/api/v1/employees/clients", "method": "POST", "description": "Client onboarding", "expect_error": True},
        {"path": "/api/v1/automation/connectors", "method": "POST", "description": "Automation connectors", "expect_error": True},
        {"path": "/api/v1/compliance/status", "method": "GET", "description": "Compliance status"},
        {"path": "/api/v1/integrations/connectors", "method": "POST", "description": "API integrations", "expect_error": True},
    ]
    
    results = []
    
    async with httpx.AsyncClient() as client:
        for endpoint in endpoints_to_test:
            try:
                if endpoint["method"] == "GET":
                    response = await client.get(f"http://127.0.0.1:8000{endpoint['path']}", timeout=10.0)
                elif endpoint["method"] == "POST":
                    response = await client.post(f"http://127.0.0.1:8000{endpoint['path']}", json={}, timeout=10.0)
                
                # Check if endpoint exists (not 404)
                if response.status_code == 404:
                    logger.error(f"❌ {endpoint['description']}: Endpoint not found (404)")
                    results.append(False)
                elif endpoint.get("expect_error") and response.status_code in [400, 422]:
                    logger.info(f"✅ {endpoint['description']}: Endpoint exists (expected error {response.status_code})")
                    results.append(True)
                elif not endpoint.get("expect_error") and response.status_code == 200:
                    logger.info(f"✅ {endpoint['description']}: Endpoint working (200)")
                    results.append(True)
                elif not endpoint.get("expect_error"):
                    logger.warning(f"⚠️  {endpoint['description']}: Unexpected status {response.status_code}")
                    results.append(True)  # Still counts as working if not 404
                else:
                    logger.info(f"✅ {endpoint['description']}: Endpoint exists (status {response.status_code})")
                    results.append(True)
                    
            except Exception as e:
                logger.error(f"❌ {endpoint['description']}: Error - {e}")
                results.append(False)
    
    return results

async def test_openapi_spec():
    """Test that OpenAPI specification is generated correctly"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://127.0.0.1:8000/openapi.json", timeout=10.0)
            
            if response.status_code == 200:
                openapi_spec = response.json()
                
                # Check for expected paths
                paths = openapi_spec.get("paths", {})
                expected_prefixes = [
                    "/api/v1/employees",
                    "/api/v1/automation", 
                    "/api/v1/compliance",
                    "/api/v1/integrations"
                ]
                
                found_prefixes = []
                for path in paths.keys():
                    for prefix in expected_prefixes:
                        if path.startswith(prefix) and prefix not in found_prefixes:
                            found_prefixes.append(prefix)
                
                logger.info(f"✅ OpenAPI spec generated successfully")
                logger.info(f"   Total endpoints: {len(paths)}")
                logger.info(f"   API prefixes found: {found_prefixes}")
                
                missing_prefixes = set(expected_prefixes) - set(found_prefixes)
                if missing_prefixes:
                    logger.warning(f"⚠️  Missing API prefixes: {missing_prefixes}")
                
                return len(missing_prefixes) == 0
            else:
                logger.error(f"❌ Failed to get OpenAPI spec: {response.status_code}")
                return False
                
    except Exception as e:
        logger.error(f"❌ OpenAPI spec test error: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("🚀 Starting API Server Startup Test")
    logger.info("=" * 50)
    
    # Start server in background process
    logger.info("Starting FastAPI server...")
    server_process = Process(target=start_server)
    server_process.start()
    
    try:
        # Wait for server to start
        logger.info("Waiting for server to start...")
        server_started = await test_server_startup()
        
        if not server_started:
            logger.error("❌ Server failed to start")
            return False
        
        # Test API endpoints
        logger.info("\n📋 Testing API endpoint registration...")
        endpoint_results = await test_api_endpoints()
        
        # Test OpenAPI specification
        logger.info("\n📋 Testing OpenAPI specification...")
        openapi_result = await test_openapi_spec()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 STARTUP TEST RESULTS")
        logger.info("=" * 50)
        
        passed_endpoints = sum(endpoint_results)
        total_endpoints = len(endpoint_results)
        
        logger.info(f"Server Startup: ✅ SUCCESS")
        logger.info(f"API Endpoints: {passed_endpoints}/{total_endpoints} working ({passed_endpoints/total_endpoints*100:.1f}%)")
        logger.info(f"OpenAPI Spec: {'✅ SUCCESS' if openapi_result else '❌ FAILED'}")
        
        overall_success = server_started and all(endpoint_results) and openapi_result
        
        if overall_success:
            logger.info("\n🎉 All startup tests passed! API server is ready.")
        else:
            logger.warning("\n⚠️  Some startup tests failed. Check logs above.")
        
        return overall_success
        
    finally:
        # Clean up server process
        logger.info("\nShutting down test server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        sys.exit(1)