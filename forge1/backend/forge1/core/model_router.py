"""
Forge 1 Enhanced Model Router
Multi-model AI routing with intelligent selection and optimization
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported AI model types"""
    GPT4O = "gpt-4o"
    GPT4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    GEMINI_PRO = "gemini-pro"
    GEMINI_ULTRA = "gemini-ultra"

class ModelCapability(Enum):
    """Model capabilities"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"

class ModelRouter:
    """Enhanced model router with intelligent selection"""
    
    def __init__(self):
        self.models = {
            ModelType.GPT4O: {
                "name": "GPT-4o",
                "provider": "openai",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.REASONING,
                    ModelCapability.MULTIMODAL
                ],
                "max_tokens": 128000,
                "cost_per_token": 0.00001,
                "latency_ms": 800,
                "available": True,
                "performance_score": 0.95
            },
            ModelType.CLAUDE_3_OPUS: {
                "name": "Claude 3 Opus",
                "provider": "anthropic", 
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.REASONING
                ],
                "max_tokens": 200000,
                "cost_per_token": 0.000015,
                "latency_ms": 1200,
                "available": True,
                "performance_score": 0.93
            },
            ModelType.GEMINI_PRO: {
                "name": "Gemini Pro",
                "provider": "google",
                "capabilities": [
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.ANALYSIS,
                    ModelCapability.MULTIMODAL
                ],
                "max_tokens": 32000,
                "cost_per_token": 0.000005,
                "latency_ms": 600,
                "available": True,
                "performance_score": 0.88
            }
        }
        
        self.active_clients = {}
        self.usage_stats = {}
        
    async def get_optimal_client(self, task_input):
        """Get optimal model client based on task requirements"""
        try:
            # Analyze task complexity and requirements
            task_complexity = await self._analyze_task_complexity(task_input)
            required_capabilities = await self._determine_required_capabilities(task_input)
            
            # Select best model
            selected_model = await self._select_optimal_model(
                task_complexity, required_capabilities
            )
            
            # Get or create client
            client = await self._get_or_create_client(selected_model)
            
            logger.info(f"Selected model {selected_model.value} for task")
            return client
            
        except Exception as e:
            logger.error(f"Error selecting optimal client: {e}")
            # Fallback to default model
            return await self._get_or_create_client(ModelType.GPT4O)
    
    async def _analyze_task_complexity(self, task_input) -> float:
        """Analyze task complexity (0.0 to 1.0)"""
        if hasattr(task_input, 'description'):
            description = task_input.description
        elif isinstance(task_input, str):
            description = task_input
        else:
            description = str(task_input)
        
        # Simple complexity analysis
        complexity_indicators = [
            ("code", 0.3),
            ("analysis", 0.4),
            ("complex", 0.5),
            ("multi-step", 0.6),
            ("enterprise", 0.7),
            ("integration", 0.8)
        ]
        
        complexity = 0.2  # Base complexity
        for indicator, weight in complexity_indicators:
            if indicator in description.lower():
                complexity += weight
        
        return min(complexity, 1.0)
    
    async def _determine_required_capabilities(self, task_input) -> List[ModelCapability]:
        """Determine required model capabilities"""
        if hasattr(task_input, 'description'):
            description = task_input.description.lower()
        elif isinstance(task_input, str):
            description = task_input.lower()
        else:
            description = str(task_input).lower()
        
        capabilities = [ModelCapability.TEXT_GENERATION]  # Always needed
        
        if any(term in description for term in ["code", "programming", "script", "function"]):
            capabilities.append(ModelCapability.CODE_GENERATION)
        
        if any(term in description for term in ["analyze", "analysis", "data", "metrics"]):
            capabilities.append(ModelCapability.ANALYSIS)
        
        if any(term in description for term in ["reason", "logic", "complex", "solve"]):
            capabilities.append(ModelCapability.REASONING)
        
        if any(term in description for term in ["image", "visual", "multimodal"]):
            capabilities.append(ModelCapability.MULTIMODAL)
        
        return capabilities
    
    async def _select_optimal_model(self, complexity: float, 
                                  capabilities: List[ModelCapability]) -> ModelType:
        """Select optimal model based on requirements"""
        
        # Score each model
        model_scores = {}
        
        for model_type, model_info in self.models.items():
            if not model_info["available"]:
                continue
            
            score = 0.0
            
            # Capability match score
            model_capabilities = set(model_info["capabilities"])
            required_capabilities = set(capabilities)
            capability_match = len(model_capabilities.intersection(required_capabilities))
            capability_score = capability_match / len(required_capabilities)
            score += capability_score * 0.4
            
            # Performance score
            score += model_info["performance_score"] * 0.3
            
            # Cost efficiency (inverse of cost)
            cost_efficiency = 1.0 - min(model_info["cost_per_token"] * 100000, 1.0)
            score += cost_efficiency * 0.2
            
            # Latency score (inverse of latency)
            latency_score = 1.0 - min(model_info["latency_ms"] / 2000, 1.0)
            score += latency_score * 0.1
            
            model_scores[model_type] = score
        
        # Select highest scoring model
        if model_scores:
            best_model = max(model_scores.items(), key=lambda x: x[1])[0]
            return best_model
        
        # Fallback
        return ModelType.GPT4O
    
    async def _get_or_create_client(self, model_type: ModelType):
        """Get or create model client"""
        if model_type in self.active_clients:
            return self.active_clients[model_type]
        
        # Create mock client
        client = MockModelClient(model_type, self.models[model_type])
        self.active_clients[model_type] = client
        
        return client
    
    async def release_client(self, client):
        """Release model client"""
        if hasattr(client, 'model_type'):
            logger.info(f"Released client for {client.model_type.value}")
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        for model_type, model_info in self.models.items():
            models.append({
                "id": model_type.value,
                "name": model_info["name"],
                "provider": model_info["provider"],
                "capabilities": [cap.value for cap in model_info["capabilities"]],
                "max_tokens": model_info["max_tokens"],
                "available": model_info["available"],
                "performance_score": model_info["performance_score"]
            })
        return models
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for model router"""
        available_models = sum(1 for model in self.models.values() if model["available"])
        total_models = len(self.models)
        
        return {
            "status": "healthy" if available_models > 0 else "unhealthy",
            "available_models": available_models,
            "total_models": total_models,
            "active_clients": len(self.active_clients),
            "timestamp": time.time()
        }

class MockModelClient:
    """Mock model client for testing"""
    
    def __init__(self, model_type: ModelType, model_info: Dict[str, Any]):
        self.model_type = model_type
        self.model_name = model_info["name"]
        self.model_info = model_info
        self.created_at = time.time()
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response"""
        # Simulate processing time
        await asyncio.sleep(self.model_info["latency_ms"] / 1000)
        
        return f"Generated response from {self.model_name} for: {prompt[:50]}..."
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat completion"""
        await asyncio.sleep(self.model_info["latency_ms"] / 1000)
        
        last_message = messages[-1]["content"] if messages else "Hello"
        return f"Chat response from {self.model_name} for: {last_message[:50]}..."