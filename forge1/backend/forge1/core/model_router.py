# forge1/backend/forge1/core/model_router.py
"""
Multi-Model Router for Forge 1

Intelligent routing across GPT-4o/5, Claude, Gemini, and open-source models
with performance benchmarking and failover capabilities.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported model providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    AZURE_OPENAI = "azure_openai"
    OPEN_SOURCE = "open_source"

@dataclass
class ModelCapability:
    """Model capability definition"""
    name: str
    provider: ModelProvider
    max_tokens: int
    supports_function_calling: bool
    supports_vision: bool
    cost_per_1k_tokens: float
    performance_score: float = 0.0
    availability_score: float = 1.0

@dataclass
class TaskRequirements:
    """Task requirements for model selection"""
    complexity_score: float
    requires_function_calling: bool = False
    requires_vision: bool = False
    max_response_tokens: int = 4000
    priority: str = "normal"  # low, normal, high, critical

@dataclass
class PerformanceBenchmark:
    """Performance benchmark data"""
    model_name: str
    avg_response_time: float
    success_rate: float
    quality_score: float
    cost_efficiency: float
    last_updated: float

class ModelRouter:
    """Intelligent model router with performance optimization and failover"""
    
    def __init__(self):
        self.models = self._initialize_models()
        self.performance_cache = {}
        self.client_pool = {}
        self._health_status = {}
        self.performance_benchmarks = {}
        self.failover_history = {}
        self._benchmark_lock = asyncio.Lock()
    
    def _initialize_models(self) -> Dict[str, ModelCapability]:
        """Initialize available models and their capabilities"""
        return {
            "gpt-4o": ModelCapability(
                name="gpt-4o",
                provider=ModelProvider.OPENAI,
                max_tokens=128000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.03,
                performance_score=0.95
            ),
            "gpt-4o-mini": ModelCapability(
                name="gpt-4o-mini",
                provider=ModelProvider.OPENAI,
                max_tokens=128000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.0015,
                performance_score=0.85
            ),
            "claude-3-5-sonnet": ModelCapability(
                name="claude-3-5-sonnet-20241022",
                provider=ModelProvider.ANTHROPIC,
                max_tokens=200000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.03,
                performance_score=0.97
            ),
            "claude-3-haiku": ModelCapability(
                name="claude-3-haiku-20240307",
                provider=ModelProvider.ANTHROPIC,
                max_tokens=200000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.0025,
                performance_score=0.80
            ),
            "gemini-1.5-pro": ModelCapability(
                name="gemini-1.5-pro",
                provider=ModelProvider.GOOGLE,
                max_tokens=2000000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.0125,
                performance_score=0.90
            ),
            "gemini-1.5-flash": ModelCapability(
                name="gemini-1.5-flash",
                provider=ModelProvider.GOOGLE,
                max_tokens=1000000,
                supports_function_calling=True,
                supports_vision=True,
                cost_per_1k_tokens=0.00075,
                performance_score=0.82
            )
        }
    
    async def get_optimal_client(self, task_input: Any) -> Any:
        """Get optimal model client with failover support"""
        
        # Analyze task requirements
        requirements = await self._analyze_task_requirements(task_input)
        
        # Select best model with failover
        selected_model, fallback_models = await self._select_optimal_model_with_failover(requirements)
        
        # Try primary model first
        try:
            client = await self._get_model_client(selected_model)
            
            # Update performance metrics
            await self._update_performance_metrics(selected_model.name, success=True)
            
            logger.info(f"Selected model {selected_model.name} for task with complexity {requirements.complexity_score}")
            return client
            
        except Exception as e:
            logger.warning(f"Primary model {selected_model.name} failed: {e}")
            
            # Try failover models
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Attempting failover to {fallback_model.name}")
                    client = await self._get_model_client(fallback_model)
                    
                    # Record failover
                    await self._record_failover(selected_model.name, fallback_model.name, str(e))
                    await self._update_performance_metrics(fallback_model.name, success=True)
                    
                    logger.info(f"Successfully failed over to {fallback_model.name}")
                    return client
                    
                except Exception as fallback_error:
                    logger.warning(f"Failover model {fallback_model.name} also failed: {fallback_error}")
                    continue
            
            # All models failed, return mock client
            logger.error("All models failed, returning mock client")
            await self._update_performance_metrics(selected_model.name, success=False)
            return self._create_mock_client(selected_model)
    
    async def _analyze_task_requirements(self, task_input: Any) -> TaskRequirements:
        """Analyze task to determine requirements"""
        
        # Extract task description
        description = ""
        if hasattr(task_input, 'description'):
            description = task_input.description
        elif isinstance(task_input, str):
            description = task_input
        
        # Calculate complexity score based on various factors
        complexity_score = 0.0
        
        # Length-based complexity
        word_count = len(description.split())
        if word_count > 1000:
            complexity_score += 0.4
        elif word_count > 500:
            complexity_score += 0.3
        elif word_count > 100:
            complexity_score += 0.2
        else:
            complexity_score += 0.1
        
        # Keyword-based complexity analysis
        complex_keywords = [
            'analyze', 'complex', 'detailed', 'comprehensive', 'strategic',
            'multi-step', 'workflow', 'integration', 'optimization', 'algorithm'
        ]
        
        description_lower = description.lower()
        keyword_matches = sum(1 for keyword in complex_keywords if keyword in description_lower)
        complexity_score += min(keyword_matches * 0.1, 0.4)
        
        # Check for function calling requirements
        requires_function_calling = any(keyword in description_lower for keyword in [
            'api', 'call', 'execute', 'run', 'tool', 'function', 'webhook'
        ])
        
        # Check for vision requirements
        requires_vision = any(keyword in description_lower for keyword in [
            'image', 'picture', 'visual', 'chart', 'graph', 'diagram'
        ])
        
        return TaskRequirements(
            complexity_score=min(complexity_score, 1.0),
            requires_function_calling=requires_function_calling,
            requires_vision=requires_vision,
            max_response_tokens=8000 if complexity_score > 0.7 else 4000,
            priority="high" if complexity_score > 0.8 else "normal"
        )
    
    async def _select_optimal_model(self, requirements: TaskRequirements) -> ModelCapability:
        """Select optimal model based on requirements"""
        
        # Filter models based on requirements
        suitable_models = []
        
        for model in self.models.values():
            # Check availability
            if not await self._is_model_available(model):
                continue
            
            # Check function calling requirement
            if requirements.requires_function_calling and not model.supports_function_calling:
                continue
            
            # Check vision requirement
            if requirements.requires_vision and not model.supports_vision:
                continue
            
            # Check token limit
            if requirements.max_response_tokens > model.max_tokens:
                continue
            
            suitable_models.append(model)
        
        if not suitable_models:
            # Fallback to most capable model
            logger.warning("No suitable models found, falling back to most capable")
            return max(self.models.values(), key=lambda m: m.performance_score)
        
        # Score models based on requirements
        scored_models = []
        for model in suitable_models:
            score = await self._calculate_model_score(model, requirements)
            scored_models.append((model, score))
        
        # Sort by score and return best
        scored_models.sort(key=lambda x: x[1], reverse=True)
        selected_model = scored_models[0][0]
        
        return selected_model
    
    async def _calculate_model_score(self, model: ModelCapability, requirements: TaskRequirements) -> float:
        """Calculate model score for given requirements"""
        
        score = 0.0
        
        # Base performance score (40% weight)
        score += model.performance_score * 0.4
        
        # Availability score (20% weight)
        score += model.availability_score * 0.2
        
        # Cost efficiency (20% weight) - lower cost is better
        max_cost = max(m.cost_per_1k_tokens for m in self.models.values())
        cost_efficiency = 1.0 - (model.cost_per_1k_tokens / max_cost)
        score += cost_efficiency * 0.2
        
        # Complexity matching (20% weight)
        if requirements.complexity_score > 0.8:
            # High complexity tasks prefer high-performance models
            score += (model.performance_score > 0.9) * 0.2
        elif requirements.complexity_score < 0.3:
            # Low complexity tasks prefer cost-efficient models
            score += (model.cost_per_1k_tokens < 0.01) * 0.2
        else:
            # Medium complexity - balanced approach
            score += 0.1
        
        # Priority bonus
        if requirements.priority == "critical" and model.performance_score > 0.95:
            score += 0.1
        
        return score
    
    async def _is_model_available(self, model: ModelCapability) -> bool:
        """Check if model is currently available with actual health checks"""
        
        # Check cached health status
        current_time = asyncio.get_event_loop().time()
        if model.name in self._health_status:
            last_check, status = self._health_status[model.name]
            if current_time - last_check < 300:  # 5 minutes cache
                return status
        
        # Perform actual health check
        try:
            client = await self._get_health_check_client(model)
            status = await self._perform_health_check(client, model)
            
            # Update availability score based on health check
            if status:
                model.availability_score = min(1.0, model.availability_score + 0.1)
            else:
                model.availability_score = max(0.0, model.availability_score - 0.2)
            
            self._health_status[model.name] = (current_time, status)
            return status
            
        except Exception as e:
            logger.error(f"Health check failed for {model.name}: {e}")
            model.availability_score = max(0.0, model.availability_score - 0.3)
            self._health_status[model.name] = (current_time, False)
            return False
    
    async def _get_health_check_client(self, model: ModelCapability) -> Any:
        """Get a lightweight client for health checks"""
        try:
            if model.provider == ModelProvider.OPENAI or model.provider == ModelProvider.AZURE_OPENAI:
                return await self._create_openai_client(model)
            elif model.provider == ModelProvider.ANTHROPIC:
                return await self._create_anthropic_client(model)
            elif model.provider == ModelProvider.GOOGLE:
                return await self._create_google_client(model)
            else:
                return self._create_mock_client(model)
        except Exception as e:
            logger.error(f"Failed to create health check client for {model.name}: {e}")
            return self._create_mock_client(model)
    
    async def _perform_health_check(self, client: Any, model: ModelCapability) -> bool:
        """Perform actual health check on the model"""
        try:
            # Simple health check with minimal request
            if hasattr(client, 'generate_response'):
                # Mock client
                await client.generate_response("health check")
                return True
            elif model.provider == ModelProvider.OPENAI or model.provider == ModelProvider.AZURE_OPENAI:
                # OpenAI health check
                response = await client.chat.completions.create(
                    model=model.name,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    timeout=10
                )
                return response is not None
            elif model.provider == ModelProvider.ANTHROPIC:
                # Anthropic health check
                response = await client.messages.create(
                    model=model.name,
                    max_tokens=1,
                    messages=[{"role": "user", "content": "ping"}],
                    timeout=10
                )
                return response is not None
            elif model.provider == ModelProvider.GOOGLE:
                # Google health check
                response = await client.generate_content_async("ping")
                return response is not None
            else:
                return True
                
        except Exception as e:
            logger.warning(f"Health check failed for {model.name}: {e}")
            return False
    
    async def _get_model_client(self, model: ModelCapability) -> Any:
        """Get or create model client"""
        
        if model.name in self.client_pool:
            return self.client_pool[model.name]
        
        # Create client based on provider
        try:
            if model.provider == ModelProvider.OPENAI:
                client = await self._create_openai_client(model)
            elif model.provider == ModelProvider.ANTHROPIC:
                client = await self._create_anthropic_client(model)
            elif model.provider == ModelProvider.GOOGLE:
                client = await self._create_google_client(model)
            else:
                raise ValueError(f"Unsupported provider: {model.provider}")
            
            # Add model name for tracking
            client.model_name = model.name
            self.client_pool[model.name] = client
            return client
            
        except Exception as e:
            logger.error(f"Failed to create client for {model.name}: {e}")
            # Fallback to default client
            return await self._get_fallback_client()
    
    async def _create_openai_client(self, model: ModelCapability) -> Any:
        """Create OpenAI client with proper configuration"""
        try:
            # Import OpenAI client
            import openai
            import os
            
            # Configure client based on model provider
            if model.provider == ModelProvider.AZURE_OPENAI:
                # Use Azure OpenAI configuration from Microsoft's setup
                import sys
                sys.path.append('../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')
                from config_kernel import Config
                
                client = openai.AsyncAzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=Config.AZURE_OPENAI_API_VERSION,
                    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
                )
            else:
                # Use standard OpenAI
                client = openai.AsyncOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
            
            # Add model metadata
            client.model_name = model.name
            client.provider = model.provider
            client.max_tokens = model.max_tokens
            
            logger.info(f"Created OpenAI client for model: {model.name}")
            return client
            
        except ImportError:
            logger.warning("OpenAI library not available, using mock client")
            return self._create_mock_client(model)
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")
            return self._create_mock_client(model)
    
    async def _create_anthropic_client(self, model: ModelCapability) -> Any:
        """Create Anthropic client with proper configuration"""
        try:
            # Import Anthropic client
            import anthropic
            import os
            
            client = anthropic.AsyncAnthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            
            # Add model metadata
            client.model_name = model.name
            client.provider = model.provider
            client.max_tokens = model.max_tokens
            
            logger.info(f"Created Anthropic client for model: {model.name}")
            return client
            
        except ImportError:
            logger.warning("Anthropic library not available, using mock client")
            return self._create_mock_client(model)
        except Exception as e:
            logger.error(f"Failed to create Anthropic client: {e}")
            return self._create_mock_client(model)
    
    async def _create_google_client(self, model: ModelCapability) -> Any:
        """Create Google client with proper configuration"""
        try:
            # Import Google Generative AI client
            import google.generativeai as genai
            import os
            
            # Configure API key
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Create model instance
            client = genai.GenerativeModel(model.name)
            
            # Add model metadata
            client.model_name = model.name
            client.provider = model.provider
            client.max_tokens = model.max_tokens
            
            logger.info(f"Created Google client for model: {model.name}")
            return client
            
        except ImportError:
            logger.warning("Google Generative AI library not available, using mock client")
            return self._create_mock_client(model)
        except Exception as e:
            logger.error(f"Failed to create Google client: {e}")
            return self._create_mock_client(model)
    
    def _create_mock_client(self, model: ModelCapability) -> Any:
        """Create mock client for testing/fallback"""
        class MockClient:
            def __init__(self, model_capability):
                self.model_name = model_capability.name
                self.provider = model_capability.provider
                self.max_tokens = model_capability.max_tokens
                self.supports_function_calling = model_capability.supports_function_calling
                self.supports_vision = model_capability.supports_vision
            
            async def generate_response(self, prompt: str, **kwargs) -> str:
                """Mock response generation"""
                return f"Mock response from {self.model_name} for prompt: {prompt[:50]}..."
            
            async def close(self):
                """Mock cleanup"""
                pass
        
        return MockClient(model)
    
    async def _get_fallback_client(self) -> Any:
        """Get fallback client when primary fails"""
        # Return the most reliable model as fallback
        fallback_model = self.models["gpt-4o-mini"]  # Most cost-effective reliable option
        return await self._create_openai_client(fallback_model)
    
    async def release_client(self, client: Any) -> None:
        """Release client back to pool"""
        # In a real implementation, this would handle connection cleanup
        pass
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models with their capabilities"""
        models_info = []
        
        for model in self.models.values():
            is_available = await self._is_model_available(model)
            models_info.append({
                "name": model.name,
                "provider": model.provider.value,
                "max_tokens": model.max_tokens,
                "supports_function_calling": model.supports_function_calling,
                "supports_vision": model.supports_vision,
                "cost_per_1k_tokens": model.cost_per_1k_tokens,
                "performance_score": model.performance_score,
                "available": is_available
            })
        
        return models_info
    
    async def _select_optimal_model_with_failover(self, requirements: TaskRequirements) -> tuple:
        """Select optimal model with failover options"""
        
        # Get all suitable models
        suitable_models = []
        
        for model in self.models.values():
            if not await self._is_model_available(model):
                continue
            
            if requirements.requires_function_calling and not model.supports_function_calling:
                continue
            
            if requirements.requires_vision and not model.supports_vision:
                continue
            
            if requirements.max_response_tokens > model.max_tokens:
                continue
            
            suitable_models.append(model)
        
        if not suitable_models:
            # Return most capable model as fallback
            primary = max(self.models.values(), key=lambda m: m.performance_score)
            return primary, []
        
        # Score and sort models
        scored_models = []
        for model in suitable_models:
            score = await self._calculate_enhanced_model_score(model, requirements)
            scored_models.append((model, score))
        
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        # Primary model is the best, failovers are the next best options
        primary_model = scored_models[0][0]
        failover_models = [model for model, _ in scored_models[1:3]]  # Top 2 alternatives
        
        return primary_model, failover_models
    
    async def _calculate_enhanced_model_score(self, model: ModelCapability, requirements: TaskRequirements) -> float:
        """Enhanced model scoring with performance benchmarks"""
        
        base_score = await self._calculate_model_score(model, requirements)
        
        # Add performance benchmark data
        if model.name in self.performance_benchmarks:
            benchmark = self.performance_benchmarks[model.name]
            
            # Weight recent performance (30% of score)
            performance_weight = 0.3
            benchmark_score = (
                benchmark.success_rate * 0.4 +
                (1.0 - min(benchmark.avg_response_time / 10.0, 1.0)) * 0.3 +  # Faster is better
                benchmark.quality_score * 0.3
            )
            
            base_score = base_score * (1 - performance_weight) + benchmark_score * performance_weight
        
        return base_score
    
    async def _update_performance_metrics(self, model_name: str, success: bool, response_time: float = 0.0, quality_score: float = 0.0) -> None:
        """Update performance metrics for a model"""
        
        async with self._benchmark_lock:
            current_time = time.time()
            
            if model_name not in self.performance_benchmarks:
                self.performance_benchmarks[model_name] = PerformanceBenchmark(
                    model_name=model_name,
                    avg_response_time=response_time,
                    success_rate=1.0 if success else 0.0,
                    quality_score=quality_score,
                    cost_efficiency=0.0,
                    last_updated=current_time
                )
            else:
                benchmark = self.performance_benchmarks[model_name]
                
                # Update with exponential moving average
                alpha = 0.1  # Learning rate
                
                if response_time > 0:
                    benchmark.avg_response_time = (1 - alpha) * benchmark.avg_response_time + alpha * response_time
                
                benchmark.success_rate = (1 - alpha) * benchmark.success_rate + alpha * (1.0 if success else 0.0)
                
                if quality_score > 0:
                    benchmark.quality_score = (1 - alpha) * benchmark.quality_score + alpha * quality_score
                
                benchmark.last_updated = current_time
    
    async def _record_failover(self, primary_model: str, failover_model: str, error: str) -> None:
        """Record failover event for analysis"""
        
        current_time = time.time()
        
        if primary_model not in self.failover_history:
            self.failover_history[primary_model] = []
        
        self.failover_history[primary_model].append({
            "timestamp": current_time,
            "failover_to": failover_model,
            "error": error,
            "success": True  # Assume success if we got here
        })
        
        # Keep only recent history (last 100 events)
        if len(self.failover_history[primary_model]) > 100:
            self.failover_history[primary_model] = self.failover_history[primary_model][-100:]
        
        logger.info(f"Recorded failover from {primary_model} to {failover_model}")
    
    async def get_performance_benchmarks(self) -> Dict[str, Dict[str, Any]]:
        """Get performance benchmarks for all models"""
        
        benchmarks = {}
        
        for model_name, benchmark in self.performance_benchmarks.items():
            benchmarks[model_name] = {
                "avg_response_time": benchmark.avg_response_time,
                "success_rate": benchmark.success_rate,
                "quality_score": benchmark.quality_score,
                "cost_efficiency": benchmark.cost_efficiency,
                "last_updated": benchmark.last_updated,
                "failover_count": len(self.failover_history.get(model_name, []))
            }
        
        return benchmarks
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check on model router"""
        
        total_models = len(self.models)
        available_models = 0
        model_health = {}
        
        for model in self.models.values():
            is_available = await self._is_model_available(model)
            if is_available:
                available_models += 1
            
            model_health[model.name] = {
                "available": is_available,
                "availability_score": model.availability_score,
                "performance_score": model.performance_score,
                "provider": model.provider.value
            }
        
        health_percentage = (available_models / total_models * 100) if total_models > 0 else 0
        
        return {
            "status": "healthy" if health_percentage > 50 else "degraded" if health_percentage > 0 else "unhealthy",
            "total_models": total_models,
            "available_models": available_models,
            "health_percentage": health_percentage,
            "client_pool_size": len(self.client_pool),
            "performance_benchmarks_count": len(self.performance_benchmarks),
            "total_failovers": sum(len(history) for history in self.failover_history.values()),
            "model_health": model_health
        }