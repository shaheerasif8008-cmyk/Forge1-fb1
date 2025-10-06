#!/usr/bin/env python3
"""
Test script for Employee Lifecycle API endpoints

This script tests the complete employee lifecycle API functionality including:
- Client onboarding
- Employee creation
- Employee interactions
- Memory management
- API integration

Requirements: 7.1, 7.2, 7.3, 7.4
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any

import httpx
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer demo_token",  # Demo token for testing
    "X-Tenant-ID": "demo_client_001",
    "X-Client-ID": "demo_client_001",
    "X-User-ID": "demo_user_001"
}

class APITester:
    """Test class for Employee Lifecycle API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.client_id = None
        self.employee_id = None
        
    async def test_health_check(self) -> bool:
        """Test API health check"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/v1/employees/health")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Health check passed: {data['status']}")
                    return True
                else:
                    logger.error(f"❌ Health check failed: {response.status_code}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
            return False
    
    async def test_client_onboarding(self) -> bool:
        """Test client onboarding"""
        try:
            client_data = {
                "name": "Demo Corporation",
                "industry": "Technology",
                "tier": "enterprise",
                "max_employees": 50,
                "allowed_models": ["gpt-4", "gpt-3.5-turbo"],
                "security_level": "high",
                "compliance_requirements": ["SOC2", "GDPR"]
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/employees/clients",
                    json=client_data,
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.client_id = data["id"]
                    logger.info(f"✅ Client onboarded: {self.client_id}")
                    logger.info(f"   Name: {data['name']}")
                    logger.info(f"   Industry: {data['industry']}")
                    logger.info(f"   Max Employees: {data['max_employees']}")
                    return True
                else:
                    logger.error(f"❌ Client onboarding failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Client onboarding error: {e}")
            return False
    
    async def test_employee_creation(self) -> bool:
        """Test employee creation"""
        try:
            if not self.client_id:
                logger.error("❌ No client ID available for employee creation")
                return False
            
            employee_requirements = {
                "role": "Customer Support Specialist",
                "industry": "Technology",
                "expertise_areas": ["customer_service", "technical_support", "product_knowledge"],
                "communication_style": "friendly",
                "tools_needed": ["email", "chat", "knowledge_base"],
                "knowledge_domains": ["product_documentation", "troubleshooting", "customer_policies"],
                "personality_traits": {
                    "empathy_level": 0.9,
                    "patience_level": 0.95,
                    "technical_depth": 0.7,
                    "response_speed": "fast"
                },
                "model_preferences": {
                    "primary_model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 1500
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/employees/clients/{self.client_id}/employees",
                    json=employee_requirements,
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    self.employee_id = data["id"]
                    logger.info(f"✅ Employee created: {self.employee_id}")
                    logger.info(f"   Name: {data['name']}")
                    logger.info(f"   Role: {data['role']}")
                    logger.info(f"   Communication Style: {data['communication_style']}")
                    return True
                else:
                    logger.error(f"❌ Employee creation failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee creation error: {e}")
            return False
    
    async def test_employee_interaction(self) -> bool:
        """Test employee interaction"""
        try:
            if not self.client_id or not self.employee_id:
                logger.error("❌ No client/employee ID available for interaction")
                return False
            
            interaction_data = {
                "message": "Hello! I need help with setting up my account. Can you guide me through the process?",
                "session_id": f"test_session_{int(time.time())}",
                "context": {
                    "user_type": "new_customer",
                    "product": "enterprise_platform",
                    "urgency": "normal"
                },
                "include_memory": True,
                "memory_limit": 10
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/v1/employees/clients/{self.client_id}/employees/{self.employee_id}/interact",
                    json=interaction_data,
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Employee interaction successful")
                    logger.info(f"   Employee: {data.get('employee_name', 'Unknown')}")
                    logger.info(f"   Response: {data['message'][:100]}...")
                    logger.info(f"   Model Used: {data.get('model_used', 'Unknown')}")
                    logger.info(f"   Processing Time: {data.get('processing_time_ms', 0)}ms")
                    return True
                else:
                    logger.error(f"❌ Employee interaction failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee interaction error: {e}")
            return False
    
    async def test_employee_memory(self) -> bool:
        """Test employee memory retrieval"""
        try:
            if not self.client_id or not self.employee_id:
                logger.error("❌ No client/employee ID available for memory test")
                return False
            
            async with httpx.AsyncClient() as client:
                # Test recent memory retrieval
                response = await client.get(
                    f"{self.base_url}/api/v1/employees/clients/{self.client_id}/employees/{self.employee_id}/memory",
                    params={"limit": 5},
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Memory retrieval successful")
                    logger.info(f"   Memory Count: {data['total_count']}")
                    logger.info(f"   Query Time: {data['query_time_ms']:.2f}ms")
                    
                    if data['memories']:
                        latest_memory = data['memories'][0]
                        logger.info(f"   Latest Memory: {latest_memory['content'][:50]}...")
                    
                    return True
                else:
                    logger.error(f"❌ Memory retrieval failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Memory retrieval error: {e}")
            return False
    
    async def test_employee_search(self) -> bool:
        """Test employee search functionality"""
        try:
            if not self.client_id:
                logger.error("❌ No client ID available for employee search")
                return False
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/employees/clients/{self.client_id}/employees/search",
                    params={"query": "customer support", "limit": 10},
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Employee search successful")
                    logger.info(f"   Results Count: {data['total_count']}")
                    logger.info(f"   Query: {data['query']}")
                    return True
                else:
                    logger.error(f"❌ Employee search failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee search error: {e}")
            return False
    
    async def test_employee_stats(self) -> bool:
        """Test employee statistics"""
        try:
            if not self.client_id or not self.employee_id:
                logger.error("❌ No client/employee ID available for stats test")
                return False
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/v1/employees/clients/{self.client_id}/employees/{self.employee_id}/stats",
                    headers=HEADERS
                )
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✅ Employee stats retrieval successful")
                    logger.info(f"   Total Interactions: {data.get('total_interactions', 0)}")
                    logger.info(f"   Average Response Time: {data.get('avg_response_time_ms', 0):.2f}ms")
                    return True
                else:
                    logger.error(f"❌ Employee stats failed: {response.status_code}")
                    logger.error(f"   Response: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Employee stats error: {e}")
            return False
    
    async def run_all_tests(self) -> Dict[str, bool]:
        """Run all API tests"""
        logger.info("🚀 Starting Employee Lifecycle API Tests")
        logger.info("=" * 50)
        
        results = {}
        
        # Test 1: Health Check
        logger.info("\n📋 Test 1: Health Check")
        results["health_check"] = await self.test_health_check()
        
        # Test 2: Client Onboarding
        logger.info("\n📋 Test 2: Client Onboarding")
        results["client_onboarding"] = await self.test_client_onboarding()
        
        # Test 3: Employee Creation
        logger.info("\n📋 Test 3: Employee Creation")
        results["employee_creation"] = await self.test_employee_creation()
        
        # Test 4: Employee Interaction
        logger.info("\n📋 Test 4: Employee Interaction")
        results["employee_interaction"] = await self.test_employee_interaction()
        
        # Test 5: Employee Memory
        logger.info("\n📋 Test 5: Employee Memory")
        results["employee_memory"] = await self.test_employee_memory()
        
        # Test 6: Employee Search
        logger.info("\n📋 Test 6: Employee Search")
        results["employee_search"] = await self.test_employee_search()
        
        # Test 7: Employee Stats
        logger.info("\n📋 Test 7: Employee Statistics")
        results["employee_stats"] = await self.test_employee_stats()
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("📊 TEST RESULTS SUMMARY")
        logger.info("=" * 50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All tests passed! Employee Lifecycle API is working correctly.")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed. Please check the logs above.")
        
        return results


async def main():
    """Main test function"""
    tester = APITester()
    results = await tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())