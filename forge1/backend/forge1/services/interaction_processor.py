# forge1/backend/forge1/services/interaction_processor.py
"""
Employee Interaction Processing Service

Handles AI employee interactions with personality application, memory integration,
and response generation using the configured model preferences.

Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 6.1, 6.2
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from forge1.core.database_config import DatabaseManager
from forge1.core.model_router import ModelRouter
from forge1.core.tenancy import get_current_tenant
from forge1.services.employee_manager import EmployeeManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import (
    Employee, EmployeeInteraction, EmployeeResponse, MemoryItem,
    MemoryType, CommunicationStyle, FormalityLevel, ExpertiseLevel,
    ResponseLength, EmployeeNotFoundError, InvalidConfigurationError,
    TenantIsolationError
)

logger = logging.getLogger(__name__)


class InteractionProcessor:
    """
    Processes AI employee interactions with full personality and memory integration.
    
    Features:
    - Personality-driven response generation
    - Memory context integration
    - Model preference application
    - Tool usage coordination
    - Performance tracking
    """
    
    def __init__(
        self,
        employee_manager: Optional[EmployeeManager] = None,
        memory_manager: Optional[EmployeeMemoryManager] = None,
        model_router: Optional[ModelRouter] = None,
        db_manager: Optional[DatabaseManager] = None
    ):
        self.employee_manager = employee_manager
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.db_manager = db_manager
        self._initialized = False
        
        # Performance tracking
        self._metrics = {
            "interactions_processed": 0,
            "average_response_time": 0.0,
            "total_tokens_used": 0,
            "total_cost": Decimal('0.00'),
            "model_usage": {},
            "personality_applications": 0,
            "memory_retrievals": 0
        }
    
    async def initialize(self):
        """Initialize the interaction processor"""
        if self._initialized:
            return
        
        # Initialize dependencies if not provided
        if not self.employee_manager:
            self.employee_manager = EmployeeManager()
            await self.employee_manager.initialize()
        
        if not self.memory_manager:
            self.memory_manager = self.employee_manager.memory_manager
        
        if not self.model_router:
            self.model_router = self.employee_manager.model_router
        
        if not self.db_manager:
            self.db_manager = self.employee_manager.db_manager
        
        self._initialized = True
        logger.info("Interaction Processor initialized")
    
    async def process_interaction(
        self,
        client_id: str,
        employee_id: str,
        message: str,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        include_memory: bool = True,
        memory_limit: int = 10
    ) -> EmployeeResponse:
        """
        Process a complete interaction with an AI employee.
        
        This is the main entry point for employee interactions, handling:
        1. Employee loading with memory context
        2. Personality application
        3. AI response generation
        4. Memory storage
        5. Performance tracking
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        interaction_id = str(uuid.uuid4())
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Load employee with memory context
            employee = await self.employee_manager.load_employee(
                client_id, employee_id,
                include_memory_context=include_memory,
                memory_limit=memory_limit
            )
            
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Create interaction record
            interaction = EmployeeInteraction(
                id=interaction_id,
                employee_id=employee_id,
                client_id=client_id,
                session_id=session_id or f"session_{uuid.uuid4().hex[:8]}",
                message=message,
                context=context or {},
                timestamp=datetime.now(timezone.utc)
            )
            
            # Build context for AI processing
            ai_context = await self._build_ai_context(employee, interaction)
            
            # Generate AI response
            response = await self._generate_ai_response(employee, interaction, ai_context)
            
            # Store interaction in memory
            await self.memory_manager.store_interaction(
                client_id, employee_id, interaction, response
            )
            
            # Update employee last interaction
            await self._update_employee_interaction_timestamp(client_id, employee_id)
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_metrics(response, processing_time)
            
            logger.info(f"Processed interaction {interaction_id} for employee {employee_id} in {processing_time:.3f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Interaction processing failed for employee {employee_id}: {e}")
            
            # Return error response
            return EmployeeResponse(
                message=f"I apologize, but I encountered an error processing your request. Please try again.",
                employee_id=employee_id,
                interaction_id=interaction_id,
                timestamp=datetime.now(timezone.utc),
                model_used="error",
                tokens_used=0,
                processing_time_ms=processing_time * 1000,
                cost=Decimal('0.00'),
                confidence_score=0.0
            )
    
    async def _build_ai_context(
        self,
        employee: Employee,
        interaction: EmployeeInteraction
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for AI processing including personality,
        memory, tools, and knowledge sources.
        """
        context = {
            "employee": {
                "name": employee.name,
                "role": employee.role,
                "personality": self._format_personality_for_ai(employee.personality),
                "tools": employee.tool_access,
                "knowledge_sources": employee.knowledge_sources
            },
            "interaction": {
                "message": interaction.message,
                "session_id": interaction.session_id,
                "context": interaction.context,
                "timestamp": interaction.timestamp.isoformat()
            },
            "memory_context": [],
            "knowledge_context": []
        }
        
        # Add memory context if available
        if employee.memory_context:
            context["memory_context"] = [
                {
                    "content": memory.content,
                    "response": memory.response,
                    "timestamp": memory.timestamp.isoformat(),
                    "relevance": memory.relevance_score,
                    "type": memory.memory_type.value
                }
                for memory in employee.memory_context
            ]
            
            self._metrics["memory_retrievals"] += 1
        
        # Search knowledge base for relevant information
        if employee.knowledge_sources:
            try:
                knowledge_results = await self.memory_manager.search_knowledge_base(
                    interaction.client_id, employee.id,
                    query=interaction.message,
                    limit=3
                )
                
                context["knowledge_context"] = [
                    {
                        "title": kb["title"],
                        "content": kb["content"][:500],  # Truncate for context
                        "relevance": kb["relevance_score"],
                        "source": kb["source_type"]
                    }
                    for kb in knowledge_results
                ]
                
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")
        
        return context
    
    def _format_personality_for_ai(self, personality) -> Dict[str, Any]:
        """Format personality configuration for AI system prompt"""
        return {
            "communication_style": personality.communication_style.value,
            "formality_level": personality.formality_level.value,
            "expertise_level": personality.expertise_level.value,
            "response_length": personality.response_length.value,
            "creativity_level": personality.creativity_level,
            "empathy_level": personality.empathy_level,
            "custom_traits": personality.custom_traits
        }
    
    async def _generate_ai_response(
        self,
        employee: Employee,
        interaction: EmployeeInteraction,
        ai_context: Dict[str, Any]
    ) -> EmployeeResponse:
        """
        Generate AI response using employee's model preferences and personality.
        """
        start_time = time.time()
        
        try:
            # Build system prompt with personality
            system_prompt = self._build_system_prompt(employee, ai_context)
            
            # Build user message with context
            user_message = self._build_user_message(interaction, ai_context)
            
            # Select model based on employee preferences
            model_name = await self._select_model_for_employee(employee, interaction)
            
            # Generate response using model router
            response_text = await self._call_ai_model(
                model_name, system_prompt, user_message, employee.model_preferences
            )
            
            # Calculate tokens and cost (mock implementation)
            tokens_used = len(response_text.split()) * 1.3  # Rough estimate
            cost = self._calculate_cost(model_name, tokens_used)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Create response object
            response = EmployeeResponse(
                message=response_text,
                employee_id=employee.id,
                interaction_id=interaction.id,
                timestamp=datetime.now(timezone.utc),
                model_used=model_name,
                tokens_used=int(tokens_used),
                processing_time_ms=processing_time_ms,
                cost=cost,
                confidence_score=0.9,  # Would be calculated based on model response
                context_used=[
                    f"memory_items:{len(ai_context.get('memory_context', []))}",
                    f"knowledge_items:{len(ai_context.get('knowledge_context', []))}"
                ],
                tools_used=self._extract_tools_used(response_text, employee.tool_access)
            )
            
            # Update interaction with response details
            interaction.processing_time_ms = processing_time_ms
            interaction.model_used = model_name
            interaction.tokens_used = int(tokens_used)
            interaction.cost = cost
            
            return response
            
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            
            # Return fallback response
            fallback_message = self._generate_fallback_response(employee, interaction)
            
            return EmployeeResponse(
                message=fallback_message,
                employee_id=employee.id,
                interaction_id=interaction.id,
                timestamp=datetime.now(timezone.utc),
                model_used="fallback",
                tokens_used=len(fallback_message.split()),
                processing_time_ms=(time.time() - start_time) * 1000,
                cost=Decimal('0.00'),
                confidence_score=0.5
            )
    
    def _build_system_prompt(self, employee: Employee, ai_context: Dict[str, Any]) -> str:
        """Build comprehensive system prompt with personality and context"""
        
        personality = ai_context["employee"]["personality"]
        
        # Base system prompt
        system_prompt = f"""You are {employee.name}, a {employee.role}.

PERSONALITY & COMMUNICATION:
- Communication Style: {personality['communication_style']} 
- Formality Level: {personality['formality_level']}
- Expertise Level: {personality['expertise_level']}
- Response Length: {personality['response_length']}
- Creativity Level: {personality['creativity_level']}/1.0
- Empathy Level: {personality['empathy_level']}/1.0"""

        # Add custom traits
        if personality.get('custom_traits'):
            system_prompt += f"\n- Additional Traits: {', '.join(f'{k}: {v}' for k, v in personality['custom_traits'].items())}"
        
        # Add available tools
        if employee.tool_access:
            system_prompt += f"\n\nAVAILABLE TOOLS:\n{', '.join(employee.tool_access)}"
        
        # Add knowledge sources
        if employee.knowledge_sources:
            system_prompt += f"\n\nKNOWLEDGE AREAS:\n{', '.join(employee.knowledge_sources)}"
        
        # Add memory context if available
        memory_context = ai_context.get("memory_context", [])
        if memory_context:
            system_prompt += f"\n\nRECENT CONVERSATION CONTEXT:"
            for i, memory in enumerate(memory_context[:3], 1):  # Limit to 3 most recent
                system_prompt += f"\n{i}. User: {memory['content'][:100]}..."
                system_prompt += f"\n   You: {memory['response'][:100]}..."
        
        # Add knowledge context if available
        knowledge_context = ai_context.get("knowledge_context", [])
        if knowledge_context:
            system_prompt += f"\n\nRELEVANT KNOWLEDGE:"
            for i, kb in enumerate(knowledge_context, 1):
                system_prompt += f"\n{i}. {kb['title']}: {kb['content'][:200]}..."
        
        # Behavioral instructions based on personality
        system_prompt += f"\n\nBEHAVIORAL GUIDELINES:"
        
        if personality['communication_style'] == 'professional':
            system_prompt += "\n- Maintain professional tone and language"
        elif personality['communication_style'] == 'friendly':
            system_prompt += "\n- Be warm, approachable, and personable"
        elif personality['communication_style'] == 'technical':
            system_prompt += "\n- Use precise technical language and detailed explanations"
        
        if personality['formality_level'] == 'formal':
            system_prompt += "\n- Use formal language and proper titles"
        elif personality['formality_level'] == 'casual':
            system_prompt += "\n- Use conversational, relaxed language"
        
        if personality['response_length'] == 'concise':
            system_prompt += "\n- Keep responses brief and to the point"
        elif personality['response_length'] == 'detailed':
            system_prompt += "\n- Provide comprehensive, thorough responses"
        
        system_prompt += f"\n\nAlways respond as {employee.name} would, maintaining consistency with your personality and expertise."
        
        return system_prompt
    
    def _build_user_message(self, interaction: EmployeeInteraction, ai_context: Dict[str, Any]) -> str:
        """Build user message with additional context"""
        
        message = interaction.message
        
        # Add session context if available
        if interaction.context:
            context_items = []
            for key, value in interaction.context.items():
                if key not in ['internal', 'system']:  # Skip internal context
                    context_items.append(f"{key}: {value}")
            
            if context_items:
                message += f"\n\nContext: {', '.join(context_items)}"
        
        return message
    
    async def _select_model_for_employee(self, employee: Employee, interaction: EmployeeInteraction) -> str:
        """Select appropriate model based on employee preferences and task"""
        
        # Check for specialized models based on task type
        message_lower = interaction.message.lower()
        
        for task_type, model in employee.model_preferences.specialized_models.items():
            if task_type.lower() in message_lower:
                return model
        
        # Use primary model
        return employee.model_preferences.primary_model
    
    async def _call_ai_model(
        self,
        model_name: str,
        system_prompt: str,
        user_message: str,
        model_preferences
    ) -> str:
        """Call AI model with employee preferences"""
        
        try:
            # Use model router to get appropriate client
            # This is a simplified implementation - in production would use actual model routing
            
            # Mock response generation based on model and preferences
            response_parts = []
            
            # Simulate model-specific behavior
            if "gpt-4" in model_name:
                response_parts.append("Based on my analysis,")
            elif "claude" in model_name:
                response_parts.append("I'd be happy to help you with this.")
            elif "gemini" in model_name:
                response_parts.append("Let me provide you with a comprehensive response.")
            
            # Add personality-influenced content
            if model_preferences.temperature > 0.8:
                response_parts.append("Here's a creative approach to consider:")
            elif model_preferences.temperature < 0.3:
                response_parts.append("Here's a structured analysis:")
            
            # Mock response content
            response_parts.append(f"This is a mock response to: {user_message[:50]}...")
            response_parts.append("In a production environment, this would be generated by the actual AI model using the system prompt and user message.")
            
            return " ".join(response_parts)
            
        except Exception as e:
            logger.error(f"Model call failed for {model_name}: {e}")
            raise
    
    def _calculate_cost(self, model_name: str, tokens_used: float) -> Decimal:
        """Calculate cost based on model and token usage"""
        
        # Mock pricing (would be actual pricing in production)
        pricing = {
            "gpt-4": Decimal('0.00003'),  # $0.03 per 1K tokens
            "gpt-3.5-turbo": Decimal('0.000002'),  # $0.002 per 1K tokens
            "claude-3-opus": Decimal('0.000015'),  # $0.015 per 1K tokens
            "claude-3-sonnet": Decimal('0.000003'),  # $0.003 per 1K tokens
            "gemini-pro": Decimal('0.0000005'),  # $0.0005 per 1K tokens
        }
        
        # Find matching model
        cost_per_token = Decimal('0.00001')  # Default
        for model_key, price in pricing.items():
            if model_key in model_name.lower():
                cost_per_token = price
                break
        
        return cost_per_token * Decimal(str(tokens_used))
    
    def _extract_tools_used(self, response_text: str, available_tools: List[str]) -> List[str]:
        """Extract which tools were mentioned/used in the response"""
        
        tools_used = []
        response_lower = response_text.lower()
        
        for tool in available_tools:
            if tool.lower() in response_lower or tool.replace('_', ' ').lower() in response_lower:
                tools_used.append(tool)
        
        return tools_used
    
    def _generate_fallback_response(self, employee: Employee, interaction: EmployeeInteraction) -> str:
        """Generate fallback response when AI model fails"""
        
        personality = employee.personality
        
        if personality.communication_style == CommunicationStyle.PROFESSIONAL:
            return f"I apologize, but I'm experiencing technical difficulties at the moment. As your {employee.role}, I want to ensure I provide you with accurate information. Please try your request again, and I'll do my best to assist you."
        elif personality.communication_style == CommunicationStyle.FRIENDLY:
            return f"Oops! I'm having a bit of trouble right now. Don't worry though - as your {employee.role}, I'm here to help! Could you please try asking me again?"
        elif personality.communication_style == CommunicationStyle.TECHNICAL:
            return f"System temporarily unavailable. As your {employee.role}, I require a moment to process your request properly. Please retry your query."
        else:
            return f"I'm currently unable to process your request. Please try again shortly."
    
    async def _update_employee_interaction_timestamp(self, client_id: str, employee_id: str):
        """Update employee's last interaction timestamp"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute("""
                    UPDATE forge1_employees.employees 
                    SET last_interaction_at = NOW(), updated_at = NOW()
                    WHERE id = $1 AND client_id = $2
                """, employee_id, client_id)
        except Exception as e:
            logger.warning(f"Failed to update employee interaction timestamp: {e}")
    
    async def _update_metrics(self, response: EmployeeResponse, processing_time: float):
        """Update interaction processing metrics"""
        
        # Update counters
        self._metrics["interactions_processed"] += 1
        self._metrics["total_tokens_used"] += response.tokens_used
        self._metrics["total_cost"] += response.cost
        self._metrics["personality_applications"] += 1
        
        # Update average response time
        current_avg = self._metrics["average_response_time"]
        interaction_count = self._metrics["interactions_processed"]
        self._metrics["average_response_time"] = (
            (current_avg * (interaction_count - 1) + processing_time) / interaction_count
        )
        
        # Update model usage
        model = response.model_used
        if model not in self._metrics["model_usage"]:
            self._metrics["model_usage"][model] = {"count": 0, "total_tokens": 0, "total_cost": Decimal('0.00')}
        
        self._metrics["model_usage"][model]["count"] += 1
        self._metrics["model_usage"][model]["total_tokens"] += response.tokens_used
        self._metrics["model_usage"][model]["total_cost"] += response.cost
    
    async def _validate_tenant_access(self, client_id: str) -> None:
        """Validate that current tenant can access the client"""
        current_tenant = get_current_tenant()
        if current_tenant and current_tenant != client_id:
            raise TenantIsolationError(
                f"Tenant {current_tenant} cannot access client {client_id}"
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get interaction processing metrics"""
        return {
            **self._metrics,
            "average_cost_per_interaction": (
                float(self._metrics["total_cost"]) / max(self._metrics["interactions_processed"], 1)
            ),
            "average_tokens_per_interaction": (
                self._metrics["total_tokens_used"] / max(self._metrics["interactions_processed"], 1)
            )
        }