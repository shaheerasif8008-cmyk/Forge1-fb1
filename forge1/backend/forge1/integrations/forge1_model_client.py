"""
Forge1 Model Client

Model client that routes all MCAE model requests through Forge1's ModelRouter,
ensuring tenant isolation, employee-specific model preferences, and proper
authentication and billing.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.model_router import ModelRouter

logger = logging.getLogger(__name__)


class Forge1ModelClient:
    """
    Model client that routes through Forge1's model router.
    
    This client ensures that all MCAE model requests go through Forge1's
    enterprise model routing system, maintaining tenant isolation and
    respecting employee model preferences.
    """
    
    def __init__(self, tenant_id: str, employee_id: str, model_router: ModelRouter):
        """
        Initialize the Forge1 model client.
        
        Args:
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for personalization
            model_router: Forge1's model router instance
        """
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.model_router = model_router
        
        # Client state
        self.created_at = time.time()
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
        # Model preferences (will be loaded from employee config)
        self.default_temperature = 0.7
        self.default_max_tokens = 2000
        self.preferred_model = None
        
    async def initialize(self, employee_config: Optional[Dict] = None):
        """
        Initialize client with employee configuration.
        
        Args:
            employee_config: Employee model preferences and configuration
        """
        if employee_config:
            model_prefs = employee_config.get('model_preferences', {})
            self.default_temperature = model_prefs.get('temperature', 0.7)
            self.default_max_tokens = model_prefs.get('max_tokens', 2000)
            self.preferred_model = model_prefs.get('primary_model')
        
        logger.debug(f"Initialized Forge1ModelClient for employee {self.employee_id}")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate response through Forge1 model router.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        return await self._make_request("generate", prompt=prompt, **kwargs)
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Chat completion through Forge1 model router.
        
        Args:
            messages: List of chat messages in OpenAI format
            **kwargs: Additional chat parameters
            
        Returns:
            Chat response text
        """
        return await self._make_request("chat", messages=messages, **kwargs)
    
    async def _make_request(self, request_type: str, **kwargs) -> str:
        """
        Make a model request through Forge1's router with proper context.
        
        Args:
            request_type: Type of request ("generate" or "chat")
            **kwargs: Request parameters
            
        Returns:
            Model response
        """
        start_time = time.time()
        
        try:
            # Set tenant context for the request
            original_tenant = get_current_tenant()
            set_current_tenant(self.tenant_id)
            
            # Prepare request parameters
            request_params = self._prepare_request_params(**kwargs)
            
            # Create task input for model router
            task_input = self._create_task_input(request_type, request_params)
            
            # Get optimal client from model router
            model_client = await self.model_router.get_optimal_client(task_input)
            
            # Make the actual request
            if request_type == "generate":
                response = await model_client.generate(
                    request_params.get("prompt", ""),
                    **self._extract_generation_params(request_params)
                )
            elif request_type == "chat":
                response = await model_client.chat(
                    request_params.get("messages", []),
                    **self._extract_chat_params(request_params)
                )
            else:
                raise ValueError(f"Unknown request type: {request_type}")
            
            # Release the client
            await self.model_router.release_client(model_client)
            
            # Update metrics
            self._update_metrics(request_params, response, time.time() - start_time)
            
            # Restore original tenant context
            if original_tenant:
                set_current_tenant(original_tenant)
            
            logger.debug(f"Model request completed for employee {self.employee_id}")
            return response
            
        except Exception as e:
            logger.error(f"Model request failed for employee {self.employee_id}: {e}")
            
            # Restore original tenant context on error
            if original_tenant:
                set_current_tenant(original_tenant)
            
            # Return a fallback response or re-raise
            raise RuntimeError(f"Model request failed: {e}")
    
    def _prepare_request_params(self, **kwargs) -> Dict[str, Any]:
        """Prepare and validate request parameters"""
        
        params = dict(kwargs)
        
        # Set defaults from employee preferences
        if 'temperature' not in params:
            params['temperature'] = self.default_temperature
        
        if 'max_tokens' not in params:
            params['max_tokens'] = self.default_max_tokens
        
        # Add tenant and employee context
        params['tenant_id'] = self.tenant_id
        params['employee_id'] = self.employee_id
        params['request_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Validate temperature
        if not 0.0 <= params['temperature'] <= 2.0:
            params['temperature'] = self.default_temperature
            logger.warning(f"Invalid temperature, using default: {self.default_temperature}")
        
        # Validate max_tokens
        if params['max_tokens'] <= 0:
            params['max_tokens'] = self.default_max_tokens
            logger.warning(f"Invalid max_tokens, using default: {self.default_max_tokens}")
        
        return params
    
    def _create_task_input(self, request_type: str, params: Dict) -> Dict:
        """Create task input for model router selection"""
        
        # Extract description for complexity analysis
        if request_type == "generate":
            description = params.get("prompt", "")
        elif request_type == "chat":
            messages = params.get("messages", [])
            description = " ".join([msg.get("content", "") for msg in messages])
        else:
            description = f"{request_type} request"
        
        return {
            "description": description,
            "request_type": request_type,
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "preferred_model": self.preferred_model,
            "parameters": params
        }
    
    def _extract_generation_params(self, params: Dict) -> Dict:
        """Extract parameters specific to generation requests"""
        generation_params = {}
        
        # Standard generation parameters
        for key in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']:
            if key in params:
                generation_params[key] = params[key]
        
        return generation_params
    
    def _extract_chat_params(self, params: Dict) -> Dict:
        """Extract parameters specific to chat requests"""
        chat_params = {}
        
        # Standard chat parameters
        for key in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'stream']:
            if key in params:
                chat_params[key] = params[key]
        
        return chat_params
    
    def _update_metrics(self, params: Dict, response: str, duration: float):
        """Update client metrics"""
        self.request_count += 1
        
        # Estimate token usage (rough approximation)
        input_text = ""
        if "prompt" in params:
            input_text = params["prompt"]
        elif "messages" in params:
            input_text = " ".join([msg.get("content", "") for msg in params["messages"]])
        
        estimated_input_tokens = len(input_text.split())
        estimated_output_tokens = len(response.split())
        total_tokens = estimated_input_tokens + estimated_output_tokens
        
        self.total_tokens += total_tokens
        
        # Estimate cost (rough approximation - $0.00001 per token)
        estimated_cost = total_tokens * 0.00001
        self.total_cost += estimated_cost
        
        logger.debug(f"Request metrics - Tokens: {total_tokens}, Duration: {duration:.3f}s, Cost: ${estimated_cost:.6f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client usage metrics"""
        uptime = time.time() - self.created_at
        
        return {
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "request_count": self.request_count,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "uptime_seconds": uptime,
            "requests_per_minute": (self.request_count / (uptime / 60)) if uptime > 0 else 0,
            "average_tokens_per_request": self.total_tokens / self.request_count if self.request_count > 0 else 0
        }
    
    def reset_metrics(self):
        """Reset client metrics"""
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.created_at = time.time()
        
        logger.info(f"Reset metrics for employee {self.employee_id}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the model client"""
        try:
            # Test basic connectivity by making a simple request
            test_response = await self.generate("Hello", max_tokens=5)
            
            return {
                "status": "healthy",
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id,
                "test_response_length": len(test_response),
                "metrics": self.get_metrics()
            }
            
        except Exception as e:
            logger.error(f"Model client health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "tenant_id": self.tenant_id,
                "employee_id": self.employee_id
            }
    
    def close(self):
        """Clean up client resources"""
        logger.info(f"Closing Forge1ModelClient for employee {self.employee_id}")
        # Any cleanup logic would go here
    
    def __str__(self) -> str:
        return f"Forge1ModelClient(tenant={self.tenant_id}, employee={self.employee_id})"
    
    def __repr__(self) -> str:
        return self.__str__()


class MockForge1ModelClient(Forge1ModelClient):
    """
    Mock version of Forge1ModelClient for testing.
    
    Provides the same interface but returns mock responses without
    making actual model API calls.
    """
    
    async def _make_request(self, request_type: str, **kwargs) -> str:
        """Mock implementation that returns test responses"""
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Update metrics
        self._update_metrics(kwargs, "mock response", 0.1)
        
        # Return mock response based on request type
        if request_type == "generate":
            prompt = kwargs.get("prompt", "")
            return f"Mock generated response for: {prompt[:50]}..."
        elif request_type == "chat":
            messages = kwargs.get("messages", [])
            last_message = messages[-1].get("content", "") if messages else "Hello"
            return f"Mock chat response for: {last_message[:50]}..."
        else:
            return f"Mock response for {request_type}"