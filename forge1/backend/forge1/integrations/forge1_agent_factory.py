"""
Forge1-Aware Agent Factory

Extends MCAE's AgentFactory to create agents that are integrated with Forge1's
enterprise features including tenancy, employee configuration, and security.
"""

import logging
import sys
import os
from typing import Dict, List, Optional, Any, Type

# Add MCAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend'))

from forge1.models.employee_models import Employee, CommunicationStyle, ExpertiseLevel
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.integrations.forge1_model_client import Forge1ModelClient
from forge1.integrations.forge1_memory_store import Forge1MemoryStore

# Import MCAE components
from kernel_agents.agent_factory import AgentFactory
from kernel_agents.agent_base import BaseAgent
from kernel_agents.generic_agent import GenericAgent
from kernel_agents.hr_agent import HrAgent
from kernel_agents.marketing_agent import MarketingAgent
from kernel_agents.planner_agent import PlannerAgent
from kernel_agents.procurement_agent import ProcurementAgent
from kernel_agents.product_agent import ProductAgent
from kernel_agents.tech_support_agent import TechSupportAgent
from kernel_agents.human_agent import HumanAgent
from kernel_agents.group_chat_manager import GroupChatManager
from models.messages_kernel import AgentType

logger = logging.getLogger(__name__)


class Forge1AgentFactory(AgentFactory):
    """
    Extended MCAE Agent Factory with Forge1 integration.
    
    Creates MCAE agents that are configured with Forge1 employee settings,
    use Forge1's model router and memory manager, and respect tenant isolation.
    """
    
    def __init__(self, tenant_id: str, employee_manager, model_router, memory_manager):
        """
        Initialize the Forge1-aware agent factory.
        
        Args:
            tenant_id: Current tenant ID for isolation
            employee_manager: Forge1's employee manager
            model_router: Forge1's model router
            memory_manager: Forge1's memory manager
        """
        super().__init__()
        self.tenant_id = tenant_id
        self.employee_manager = employee_manager
        self.model_router = model_router
        self.memory_manager = memory_manager
        
        # Agent type to role mapping for employees
        self.role_to_agent_type = {
            "hr": AgentType.HR,
            "human_resources": AgentType.HR,
            "marketing": AgentType.MARKETING,
            "product": AgentType.PRODUCT,
            "procurement": AgentType.PROCUREMENT,
            "tech_support": AgentType.TECH_SUPPORT,
            "technical_support": AgentType.TECH_SUPPORT,
            "support": AgentType.TECH_SUPPORT,
            "generic": AgentType.GENERIC,
            "general": AgentType.GENERIC,
            "planner": AgentType.PLANNER,
            "human": AgentType.HUMAN,
            "intake": AgentType.GENERIC,  # Legal intake -> Generic with custom config
            "lawyer": AgentType.GENERIC,  # Lawyer -> Generic with legal specialization
            "attorney": AgentType.GENERIC,
            "research": AgentType.GENERIC,  # Research -> Generic with research focus
            "researcher": AgentType.GENERIC
        }
        
        self._initialized = False
        
    async def initialize(self):
        """Initialize the factory"""
        if self._initialized:
            return
            
        # Set tenant context
        set_current_tenant(self.tenant_id)
        
        self._initialized = True
        logger.info(f"Forge1AgentFactory initialized for tenant {self.tenant_id}")
    
    async def create_agents_for_employee(self, employee: Employee) -> Dict[str, BaseAgent]:
        """
        Create MCAE agents configured with Forge1 employee settings.
        
        Args:
            employee: Forge1 employee configuration
            
        Returns:
            Dictionary of agent instances keyed by agent type
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Set tenant context
            set_current_tenant(employee.client_id)
            
            # Determine agent type from employee role
            agent_type = self._get_agent_type_for_role(employee.role)
            
            # Create session ID for this employee's agents
            session_id = f"employee_{employee.id}_session"
            
            # Create Forge1 memory store
            memory_store = Forge1MemoryStore(
                session_id=session_id,
                user_id=employee.id,
                tenant_id=employee.client_id,
                employee_id=employee.id,
                memory_manager=self.memory_manager
            )
            
            # Create Forge1 model client
            model_client = Forge1ModelClient(
                tenant_id=employee.client_id,
                employee_id=employee.id,
                model_router=self.model_router
            )
            
            # Generate system message based on employee configuration
            system_message = self._generate_system_message(employee, agent_type)
            
            # Create the primary agent for this employee
            primary_agent = await self._create_employee_agent(
                agent_type=agent_type,
                employee=employee,
                session_id=session_id,
                memory_store=memory_store,
                model_client=model_client,
                system_message=system_message
            )
            
            # Create supporting agents (planner, group chat manager, human)
            agents = {
                agent_type.value: primary_agent
            }
            
            # Create planner agent
            planner_agent = await self._create_employee_agent(
                agent_type=AgentType.PLANNER,
                employee=employee,
                session_id=session_id,
                memory_store=memory_store,
                model_client=model_client,
                agent_instances={agent_type.value: primary_agent}
            )
            agents[AgentType.PLANNER.value] = planner_agent
            
            # Create human agent for feedback
            human_agent = await self._create_employee_agent(
                agent_type=AgentType.HUMAN,
                employee=employee,
                session_id=session_id,
                memory_store=memory_store,
                model_client=model_client
            )
            agents[AgentType.HUMAN.value] = human_agent
            
            # Create group chat manager with all agents
            all_agent_instances = {**agents}
            group_chat_manager = await self._create_employee_agent(
                agent_type=AgentType.GROUP_CHAT_MANAGER,
                employee=employee,
                session_id=session_id,
                memory_store=memory_store,
                model_client=model_client,
                agent_instances=all_agent_instances
            )
            agents[AgentType.GROUP_CHAT_MANAGER.value] = group_chat_manager
            
            logger.info(f"Created {len(agents)} agents for employee {employee.id}")
            return agents
            
        except Exception as e:
            logger.error(f"Failed to create agents for employee {employee.id}: {e}")
            raise
    
    async def _create_employee_agent(
        self,
        agent_type: AgentType,
        employee: Employee,
        session_id: str,
        memory_store,
        model_client,
        system_message: Optional[str] = None,
        agent_instances: Optional[Dict] = None
    ) -> BaseAgent:
        """Create a single agent configured for the employee"""
        
        # Get agent class
        agent_class = self._agent_classes.get(agent_type)
        if not agent_class:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        # Generate system message if not provided
        if not system_message:
            system_message = self._generate_system_message(employee, agent_type)
        
        # Prepare agent configuration
        agent_config = {
            "agent_name": f"{employee.name}_{agent_type.value}",
            "session_id": session_id,
            "user_id": employee.id,
            "memory_store": memory_store,
            "system_message": system_message,
            "client": model_client,
            "temperature": employee.model_preferences.temperature
        }
        
        # Add agent instances for planner and group chat manager
        if agent_type in [AgentType.PLANNER, AgentType.GROUP_CHAT_MANAGER] and agent_instances:
            agent_config["agent_instances"] = agent_instances
        
        # Create the agent
        agent = await agent_class.create(**agent_config)
        
        # Configure agent with employee-specific settings
        await self._configure_agent_for_employee(agent, employee)
        
        return agent
    
    def _get_agent_type_for_role(self, role: str) -> AgentType:
        """Determine MCAE agent type from employee role"""
        role_lower = role.lower().strip()
        
        # Direct mapping
        if role_lower in self.role_to_agent_type:
            return self.role_to_agent_type[role_lower]
        
        # Fuzzy matching for common variations
        if "hr" in role_lower or "human" in role_lower:
            return AgentType.HR
        elif "market" in role_lower:
            return AgentType.MARKETING
        elif "product" in role_lower:
            return AgentType.PRODUCT
        elif "procurement" in role_lower or "purchase" in role_lower:
            return AgentType.PROCUREMENT
        elif "tech" in role_lower or "support" in role_lower:
            return AgentType.TECH_SUPPORT
        elif "legal" in role_lower or "law" in role_lower:
            return AgentType.GENERIC  # Legal roles use generic with specialization
        else:
            return AgentType.GENERIC
    
    def _generate_system_message(self, employee: Employee, agent_type: AgentType) -> str:
        """Generate system message based on employee configuration"""
        
        # Base system message from agent type
        base_message = self._agent_system_messages.get(
            agent_type,
            f"You are a helpful AI assistant specialized in {agent_type.value} tasks."
        )
        
        # Customize based on employee personality
        personality_traits = []
        
        # Communication style
        if employee.personality.communication_style == CommunicationStyle.FORMAL:
            personality_traits.append("Maintain a formal and professional tone.")
        elif employee.personality.communication_style == CommunicationStyle.CASUAL:
            personality_traits.append("Use a casual and friendly communication style.")
        elif employee.personality.communication_style == CommunicationStyle.TECHNICAL:
            personality_traits.append("Use technical language and precise terminology.")
        
        # Expertise level
        if employee.personality.expertise_level == ExpertiseLevel.EXPERT:
            personality_traits.append("Provide expert-level analysis and detailed insights.")
        elif employee.personality.expertise_level == ExpertiseLevel.INTERMEDIATE:
            personality_traits.append("Provide balanced explanations suitable for intermediate users.")
        elif employee.personality.expertise_level == ExpertiseLevel.BEGINNER:
            personality_traits.append("Explain concepts clearly for beginners.")
        
        # Creativity and empathy levels
        if employee.personality.creativity_level > 0.7:
            personality_traits.append("Be creative and innovative in your responses.")
        if employee.personality.empathy_level > 0.7:
            personality_traits.append("Show empathy and understanding in your interactions.")
        
        # Role-specific customization
        role_customization = ""
        role_lower = employee.role.lower()
        
        if "intake" in role_lower:
            role_customization = " You specialize in client intake, gathering initial information, and routing cases appropriately."
        elif "lawyer" in role_lower or "attorney" in role_lower:
            role_customization = " You are a legal professional who analyzes cases, provides legal advice, and ensures compliance with regulations."
        elif "research" in role_lower:
            role_customization = " You specialize in research, finding relevant information, precedents, and supporting documentation."
        
        # Combine all elements
        system_message_parts = [base_message]
        
        if role_customization:
            system_message_parts.append(role_customization)
        
        if personality_traits:
            system_message_parts.append(" " + " ".join(personality_traits))
        
        # Add tool access information
        if employee.tool_access:
            tools_info = f" You have access to the following tools: {', '.join(employee.tool_access)}."
            system_message_parts.append(tools_info)
        
        # Add tenant context
        system_message_parts.append(f" You are working for tenant {employee.client_id} and must maintain strict data isolation.")
        
        return "".join(system_message_parts)
    
    async def _configure_agent_for_employee(self, agent: BaseAgent, employee: Employee):
        """Configure agent with employee-specific settings"""
        
        # Set agent properties based on employee configuration
        if hasattr(agent, 'set_temperature'):
            agent.set_temperature(employee.model_preferences.temperature)
        
        if hasattr(agent, 'set_max_tokens'):
            agent.set_max_tokens(employee.model_preferences.max_tokens)
        
        # Configure tool access
        if hasattr(agent, 'set_available_tools'):
            agent.set_available_tools(employee.tool_access)
        
        # Set employee context
        if hasattr(agent, 'set_employee_context'):
            agent.set_employee_context({
                "employee_id": employee.id,
                "tenant_id": employee.client_id,
                "role": employee.role,
                "name": employee.name
            })
        
        logger.debug(f"Configured agent for employee {employee.id}")
    
    async def health_check(self) -> Dict:
        """Perform health check on the agent factory"""
        try:
            return {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "tenant_id": self.tenant_id,
                "supported_agent_types": [at.value for at in self._agent_classes.keys()],
                "role_mappings": len(self.role_to_agent_type)
            }
        except Exception as e:
            logger.error(f"Agent factory health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self._initialized
            }