# forge1/backend/forge1/integrations/forge1_system_adapter.py
"""
Forge1 System Integration Adapter

Integrates the new AI Employee Lifecycle system with existing Forge1 components:
- ModelRouter integration for AI model selection
- MemoryManager integration for semantic memory
- Agent factory adaptation for employee system
- Authentication and security system compatibility

Requirements: 3.1, 3.2, 4.2, 6.1
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union

from forge1.core.model_router import ModelRouter, ModelType, ModelCapability
from forge1.core.memory_manager import MemoryManager, MemoryContext, MemoryQuery
from forge1.core.security_manager import SecurityManager
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.services.employee_manager import EmployeeManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import (
    Employee, EmployeeInteraction, EmployeeResponse, PersonalityConfig, ModelPreferences
)

logger = logging.getLogger(__name__)


class EmployeeModelRouterAdapter:
    """
    Adapter that integrates EmployeeManager with existing ModelRouter.
    
    Provides employee-specific model selection based on preferences and personality.
    """
    
    def __init__(self, model_router: ModelRouter):
        self.model_router = model_router
        self.employee_model_cache = {}
        
        # Map employee model preferences to ModelRouter types
        self.model_mapping = {
            "gpt-4": ModelType.GPT4O,
            "gpt-4-turbo": ModelType.GPT4_TURBO,
            "gpt-3.5-turbo": ModelType.GPT4_TURBO,  # Fallback mapping
            "claude-3-opus": ModelType.CLAUDE_3_OPUS,
            "claude-3-sonnet": ModelType.CLAUDE_3_SONNET,
            "gemini-pro": ModelType.GEMINI_PRO,
            "gemini-ultra": ModelType.GEMINI_ULTRA
        }
    
    async def get_employee_model_client(
        self,
        employee: Employee,
        task_context: Optional[Dict[str, Any]] = None
    ):
        """Get optimal model client for an employee based on their preferences"""
        try:
            # Check cache first
            cache_key = f"{employee.id}:{employee.model_preferences.primary_model}"
            if cache_key in self.employee_model_cache:
                cached_client, cached_time = self.employee_model_cache[cache_key]
                if time.time() - cached_time < 300:  # 5 minute cache
                    return cached_client
            
            # Determine required capabilities based on employee role and task
            required_capabilities = await self._determine_employee_capabilities(
                employee, task_context
            )
            
            # Get preferred model type
            preferred_model = self._get_preferred_model_type(employee)
            
            # Check if preferred model supports required capabilities
            if await self._model_supports_capabilities(preferred_model, required_capabilities):
                model_client = await self.model_router._get_or_create_client(preferred_model)
            else:
                # Fallback to optimal model selection
                task_input = self._create_task_input(employee, task_context)
                model_client = await self.model_router.get_optimal_client(task_input)
            
            # Apply employee-specific model configuration
            model_client = await self._configure_model_for_employee(model_client, employee)
            
            # Cache the client
            self.employee_model_cache[cache_key] = (model_client, time.time())
            
            logger.debug(f"Selected model {model_client.model_type.value} for employee {employee.id}")
            return model_client
            
        except Exception as e:
            logger.error(f"Failed to get model client for employee {employee.id}: {e}")
            # Fallback to default model
            return await self.model_router._get_or_create_client(ModelType.GPT4O)
    
    async def generate_employee_response(
        self,
        employee: Employee,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response using employee-specific model and configuration"""
        try:
            # Get employee's model client
            model_client = await self.get_employee_model_client(employee, context)
            
            # Create personality-enhanced prompt
            enhanced_prompt = await self._enhance_prompt_with_personality(
                employee, message, context
            )
            
            # Generate response with employee preferences
            response = await model_client.generate(
                enhanced_prompt,
                temperature=employee.model_preferences.temperature,
                max_tokens=employee.model_preferences.max_tokens
            )
            
            # Apply personality post-processing
            final_response = await self._apply_personality_to_response(
                employee, response, context
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Failed to generate response for employee {employee.id}: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    def _get_preferred_model_type(self, employee: Employee) -> ModelType:
        """Get employee's preferred model type"""
        preferred_model = employee.model_preferences.primary_model
        return self.model_mapping.get(preferred_model, ModelType.GPT4O)
    
    async def _determine_employee_capabilities(
        self,
        employee: Employee,
        task_context: Optional[Dict[str, Any]]
    ) -> List[ModelCapability]:
        """Determine required capabilities based on employee role and task"""
        capabilities = [ModelCapability.TEXT_GENERATION]
        
        # Add capabilities based on employee role
        role_lower = employee.role.lower()
        
        if any(keyword in role_lower for keyword in ["developer", "engineer", "programmer"]):
            capabilities.append(ModelCapability.CODE_GENERATION)
        
        if any(keyword in role_lower for keyword in ["analyst", "researcher", "scientist"]):
            capabilities.append(ModelCapability.ANALYSIS)
            capabilities.append(ModelCapability.REASONING)
        
        # Add capabilities based on task context
        if task_context:
            task_type = task_context.get("task_type", "")
            if "code" in task_type or "programming" in task_type:
                capabilities.append(ModelCapability.CODE_GENERATION)
            
            if "analysis" in task_type or "research" in task_type:
                capabilities.append(ModelCapability.ANALYSIS)
                capabilities.append(ModelCapability.REASONING)
            
            if task_context.get("has_images") or task_context.get("multimodal"):
                capabilities.append(ModelCapability.MULTIMODAL)
        
        return list(set(capabilities))  # Remove duplicates
    
    async def _model_supports_capabilities(
        self,
        model_type: ModelType,
        required_capabilities: List[ModelCapability]
    ) -> bool:
        """Check if model supports required capabilities"""
        model_info = self.model_router.models.get(model_type)
        if not model_info:
            return False
        
        model_capabilities = model_info.get("capabilities", [])
        return all(cap in model_capabilities for cap in required_capabilities)
    
    def _create_task_input(
        self,
        employee: Employee,
        task_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create task input for ModelRouter selection"""
        return {
            "description": f"Employee {employee.role} task",
            "employee_id": employee.id,
            "employee_role": employee.role,
            "context": task_context or {},
            "complexity": self._estimate_task_complexity(employee, task_context)
        }
    
    def _estimate_task_complexity(
        self,
        employee: Employee,
        task_context: Optional[Dict[str, Any]]
    ) -> float:
        """Estimate task complexity for model selection"""
        base_complexity = 0.5
        
        # Adjust based on employee expertise level
        expertise_multipliers = {
            "beginner": 0.3,
            "intermediate": 0.5,
            "expert": 0.8,
            "advanced": 0.9
        }
        
        expertise = employee.personality.expertise_level.value
        complexity = base_complexity * expertise_multipliers.get(expertise, 0.5)
        
        # Adjust based on task context
        if task_context:
            if task_context.get("complex_reasoning"):
                complexity += 0.3
            if task_context.get("code_generation"):
                complexity += 0.2
            if task_context.get("multimodal"):
                complexity += 0.2
        
        return min(complexity, 1.0)
    
    async def _configure_model_for_employee(self, model_client, employee: Employee):
        """Configure model client with employee-specific settings"""
        # Apply employee model preferences
        if hasattr(model_client, 'temperature'):
            model_client.temperature = employee.model_preferences.temperature
        
        if hasattr(model_client, 'max_tokens'):
            model_client.max_tokens = employee.model_preferences.max_tokens
        
        # Add employee context to client
        model_client.employee_id = employee.id
        model_client.employee_role = employee.role
        
        return model_client
    
    async def _enhance_prompt_with_personality(
        self,
        employee: Employee,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance prompt with employee personality"""
        personality = employee.personality
        
        # Build personality context
        personality_context = f"""
You are {employee.name}, a {employee.role}. Your communication style is {personality.communication_style.value} 
with a {personality.formality_level.value} tone. Your expertise level is {personality.expertise_level.value}.
You prefer {personality.response_length.value} responses.

Personality traits:
- Creativity level: {personality.creativity_level}/1.0
- Empathy level: {personality.empathy_level}/1.0
"""
        
        # Add custom traits if available
        if personality.custom_traits:
            traits_text = ", ".join([f"{k}: {v}" for k, v in personality.custom_traits.items()])
            personality_context += f"- Additional traits: {traits_text}\n"
        
        # Combine with original message
        enhanced_prompt = f"{personality_context}\n\nUser message: {message}"
        
        return enhanced_prompt
    
    async def _apply_personality_to_response(
        self,
        employee: Employee,
        response: str,
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Apply personality-based post-processing to response"""
        # This could include:
        # - Adjusting formality level
        # - Modifying response length
        # - Adding personality-specific phrases
        # - Applying communication style adjustments
        
        # For now, return response as-is
        # In a full implementation, this would apply NLP transformations
        return response


class EmployeeMemoryAdapter:
    """
    Adapter that integrates EmployeeMemoryManager with existing MemoryManager.
    
    Provides unified memory interface while maintaining employee isolation.
    """
    
    def __init__(
        self,
        employee_memory_manager: EmployeeMemoryManager,
        global_memory_manager: MemoryManager
    ):
        self.employee_memory = employee_memory_manager
        self.global_memory = global_memory_manager
    
    async def store_employee_interaction(
        self,
        employee: Employee,
        interaction: EmployeeInteraction,
        response: EmployeeResponse
    ):
        """Store interaction in both employee-specific and global memory"""
        try:
            # Store in employee-specific memory
            await self.employee_memory.store_interaction(
                employee.client_id,
                employee.id,
                interaction,
                response
            )
            
            # Store in global memory for cross-system compatibility
            memory_context = MemoryContext(
                content=interaction.message,
                response=response.message,
                metadata={
                    "employee_id": employee.id,
                    "client_id": employee.client_id,
                    "employee_name": employee.name,
                    "employee_role": employee.role,
                    "interaction_id": interaction.id,
                    "timestamp": interaction.timestamp.isoformat()
                },
                security_level=SecurityLevel.INTERNAL,
                memory_type=MemoryType.CONVERSATION
            )
            
            await self.global_memory.store_memory(memory_context)
            
            logger.debug(f"Stored interaction for employee {employee.id} in both memory systems")
            
        except Exception as e:
            logger.error(f"Failed to store employee interaction: {e}")
    
    async def search_employee_memory(
        self,
        employee: Employee,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search employee memory using both systems"""
        try:
            # Search employee-specific memory first
            employee_results = await self.employee_memory.search_employee_memory(
                employee.client_id,
                employee.id,
                query,
                limit
            )
            
            # If not enough results, search global memory
            if len(employee_results) < limit:
                remaining_limit = limit - len(employee_results)
                
                memory_query = MemoryQuery(
                    query=query,
                    limit=remaining_limit,
                    filters={"employee_id": employee.id}
                )
                
                global_results = await self.global_memory.search_memories(memory_query)
                
                # Convert global results to employee format
                for result in global_results.memories:
                    employee_results.append({
                        "id": result.id,
                        "content": result.content,
                        "response": result.response,
                        "timestamp": result.timestamp,
                        "relevance_score": result.relevance_score,
                        "source": "global_memory"
                    })
            
            return employee_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search employee memory: {e}")
            return []
    
    async def get_employee_context(
        self,
        employee: Employee,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent context for employee from both memory systems"""
        try:
            # Get from employee memory
            context = await self.employee_memory.get_employee_context(
                employee.client_id,
                employee.id,
                limit=limit
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to get employee context: {e}")
            return []


class EmployeeAgentFactory:
    """
    Factory for creating employee-aware agents that work with existing Forge1 systems.
    
    Adapts the existing agent factory to work with the new employee system.
    """
    
    def __init__(
        self,
        employee_manager: EmployeeManager,
        model_adapter: EmployeeModelRouterAdapter,
        memory_adapter: EmployeeMemoryAdapter,
        security_manager: SecurityManager
    ):
        self.employee_manager = employee_manager
        self.model_adapter = model_adapter
        self.memory_adapter = memory_adapter
        self.security_manager = security_manager
    
    async def create_employee_agent(
        self,
        client_id: str,
        employee_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> 'EmployeeAgent':
        """Create an agent instance for a specific employee"""
        try:
            # Load employee
            employee = await self.employee_manager.load_employee(
                client_id, employee_id, include_memory_context=True
            )
            
            if not employee:
                raise ValueError(f"Employee {employee_id} not found")
            
            # Create agent
            agent = EmployeeAgent(
                employee=employee,
                model_adapter=self.model_adapter,
                memory_adapter=self.memory_adapter,
                security_manager=self.security_manager,
                session_context=session_context or {}
            )
            
            await agent.initialize()
            
            logger.info(f"Created agent for employee {employee_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create employee agent: {e}")
            raise
    
    async def get_available_employees(self, client_id: str) -> List[Dict[str, Any]]:
        """Get list of available employees for agent creation"""
        try:
            employees = await self.employee_manager.list_employees(client_id)
            
            return [
                {
                    "id": emp.id,
                    "name": emp.name,
                    "role": emp.role,
                    "status": emp.status.value,
                    "capabilities": emp.tool_access,
                    "last_interaction": emp.last_interaction_at.isoformat() if emp.last_interaction_at else None
                }
                for emp in employees
                if emp.status.value == "active"
            ]
            
        except Exception as e:
            logger.error(f"Failed to get available employees: {e}")
            return []


class EmployeeAgent:
    """
    Employee-specific agent that integrates with existing Forge1 systems.
    
    Provides a unified interface for employee interactions while leveraging
    existing model routing, memory management, and security systems.
    """
    
    def __init__(
        self,
        employee: Employee,
        model_adapter: EmployeeModelRouterAdapter,
        memory_adapter: EmployeeMemoryAdapter,
        security_manager: SecurityManager,
        session_context: Dict[str, Any]
    ):
        self.employee = employee
        self.model_adapter = model_adapter
        self.memory_adapter = memory_adapter
        self.security_manager = security_manager
        self.session_context = session_context
        
        self.session_id = session_context.get("session_id", f"session_{int(time.time())}")
        self.interaction_count = 0
        self.created_at = datetime.now(timezone.utc)
    
    async def initialize(self):
        """Initialize the agent"""
        # Set tenant context
        set_current_tenant(self.employee.client_id)
        
        # Load recent memory context
        self.memory_context = await self.memory_adapter.get_employee_context(
            self.employee, limit=10
        )
        
        logger.debug(f"Initialized agent for employee {self.employee.id}")
    
    async def process_message(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> EmployeeResponse:
        """Process a message through the employee agent"""
        try:
            start_time = time.time()
            
            # Create interaction
            interaction = EmployeeInteraction(
                id=f"int_{self.employee.id}_{int(time.time())}",
                employee_id=self.employee.id,
                client_id=self.employee.client_id,
                message=message,
                context={
                    **self.session_context,
                    **(context or {}),
                    "session_id": self.session_id,
                    "interaction_count": self.interaction_count
                },
                timestamp=datetime.now(timezone.utc)
            )
            
            # Security check
            if not await self._security_check(message, context):
                return EmployeeResponse(
                    message="I cannot process that request due to security restrictions.",
                    employee_id=self.employee.id,
                    interaction_id=interaction.id,
                    timestamp=datetime.now(timezone.utc),
                    model_used="security_filter",
                    processing_time_ms=0,
                    tokens_used=0,
                    cost=0.0
                )
            
            # Generate response using model adapter
            response_text = await self.model_adapter.generate_employee_response(
                self.employee, message, interaction.context
            )
            
            # Create response
            processing_time = time.time() - start_time
            response = EmployeeResponse(
                message=response_text,
                employee_id=self.employee.id,
                interaction_id=interaction.id,
                timestamp=datetime.now(timezone.utc),
                model_used=self.employee.model_preferences.primary_model,
                processing_time_ms=int(processing_time * 1000),
                tokens_used=len(message.split()) + len(response_text.split()),  # Rough estimate
                cost=0.001  # Rough estimate
            )
            
            # Store in memory
            await self.memory_adapter.store_employee_interaction(
                self.employee, interaction, response
            )
            
            # Update interaction count
            self.interaction_count += 1
            
            logger.debug(f"Processed message for employee {self.employee.id} in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process message for employee {self.employee.id}: {e}")
            
            # Return error response
            return EmployeeResponse(
                message="I apologize, but I encountered an error processing your request. Please try again.",
                employee_id=self.employee.id,
                interaction_id=f"error_{int(time.time())}",
                timestamp=datetime.now(timezone.utc),
                model_used="error_handler",
                processing_time_ms=0,
                tokens_used=0,
                cost=0.0,
                error=str(e)
            )
    
    async def get_capabilities(self) -> Dict[str, Any]:
        """Get agent capabilities"""
        return {
            "employee_id": self.employee.id,
            "employee_name": self.employee.name,
            "employee_role": self.employee.role,
            "tools": self.employee.tool_access,
            "knowledge_sources": self.employee.knowledge_sources,
            "model_preferences": {
                "primary_model": self.employee.model_preferences.primary_model,
                "temperature": self.employee.model_preferences.temperature,
                "max_tokens": self.employee.model_preferences.max_tokens
            },
            "personality": {
                "communication_style": self.employee.personality.communication_style.value,
                "expertise_level": self.employee.personality.expertise_level.value,
                "formality_level": self.employee.personality.formality_level.value
            },
            "session_info": {
                "session_id": self.session_id,
                "interaction_count": self.interaction_count,
                "created_at": self.created_at.isoformat()
            }
        }
    
    async def _security_check(
        self,
        message: str,
        context: Optional[Dict[str, Any]]
    ) -> bool:
        """Perform security check on message"""
        try:
            # Use existing security manager
            # This is a simplified check - in production, use full security validation
            
            # Check for obvious security issues
            security_keywords = ["password", "secret", "token", "key", "hack", "exploit"]
            message_lower = message.lower()
            
            if any(keyword in message_lower for keyword in security_keywords):
                logger.warning(f"Security check failed for message containing sensitive keywords")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Security check error: {e}")
            return False


class Forge1SystemIntegrator:
    """
    Main integration class that coordinates all system integrations.
    
    Provides a single interface for integrating the employee lifecycle system
    with existing Forge1 components.
    """
    
    def __init__(self):
        self.model_router = None
        self.memory_manager = None
        self.security_manager = None
        self.employee_manager = None
        self.employee_memory_manager = None
        
        self.model_adapter = None
        self.memory_adapter = None
        self.agent_factory = None
        
        self._initialized = False
    
    async def initialize(
        self,
        employee_manager: EmployeeManager,
        employee_memory_manager: EmployeeMemoryManager
    ):
        """Initialize all system integrations"""
        if self._initialized:
            return
        
        try:
            # Initialize existing Forge1 systems
            self.model_router = ModelRouter()
            self.memory_manager = MemoryManager()
            self.security_manager = SecurityManager()
            
            # Store employee system references
            self.employee_manager = employee_manager
            self.employee_memory_manager = employee_memory_manager
            
            # Create adapters
            self.model_adapter = EmployeeModelRouterAdapter(self.model_router)
            self.memory_adapter = EmployeeMemoryAdapter(
                self.employee_memory_manager,
                self.memory_manager
            )
            
            # Create agent factory
            self.agent_factory = EmployeeAgentFactory(
                self.employee_manager,
                self.model_adapter,
                self.memory_adapter,
                self.security_manager
            )
            
            self._initialized = True
            logger.info("Forge1 system integration initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Forge1 system integration: {e}")
            raise
    
    async def create_employee_agent(
        self,
        client_id: str,
        employee_id: str,
        session_context: Optional[Dict[str, Any]] = None
    ) -> EmployeeAgent:
        """Create an employee agent with full system integration"""
        if not self._initialized:
            raise RuntimeError("System integration not initialized")
        
        return await self.agent_factory.create_employee_agent(
            client_id, employee_id, session_context
        )
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all system integrations"""
        return {
            "initialized": self._initialized,
            "components": {
                "model_router": self.model_router is not None,
                "memory_manager": self.memory_manager is not None,
                "security_manager": self.security_manager is not None,
                "employee_manager": self.employee_manager is not None,
                "employee_memory_manager": self.employee_memory_manager is not None
            },
            "adapters": {
                "model_adapter": self.model_adapter is not None,
                "memory_adapter": self.memory_adapter is not None,
                "agent_factory": self.agent_factory is not None
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Global integrator instance
_system_integrator = None


async def get_system_integrator() -> Forge1SystemIntegrator:
    """Get global system integrator instance"""
    global _system_integrator
    
    if _system_integrator is None:
        _system_integrator = Forge1SystemIntegrator()
    
    return _system_integrator