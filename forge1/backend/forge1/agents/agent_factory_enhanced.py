# forge1/backend/forge1/agents/agent_factory_enhanced.py
"""
Enhanced Agent Factory for Forge 1

Extends Microsoft's AgentFactory with:
- Multi-model routing capabilities
- Enhanced security and compliance
- Performance monitoring and optimization
- Framework integration (LangChain, CrewAI, AutoGen)
"""

import inspect
import logging
from typing import Any, Dict, Optional, Type

# Import Microsoft's base functionality
import sys
sys.path.append('../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend')

from kernel_agents.agent_factory import AgentFactory
from kernel_agents.agent_base import BaseAgent
from models.messages_kernel import AgentType
from context.cosmos_memory_kernel import CosmosMemoryContext

# Forge 1 Enhanced Imports
from forge1.agents.enhanced_base_agent import EnhancedBaseAgent
from forge1.agents.superhuman_planner import SuperhumanPlannerAgent
from forge1.agents.multi_model_coordinator import MultiModelCoordinator
from forge1.agents.compliance_agent import ComplianceAgent
from forge1.agents.performance_optimizer import PerformanceOptimizerAgent
from forge1.core.model_router import ModelRouter
from forge1.integrations.langchain_adapter import LangChainAdapter
from forge1.integrations.crewai_adapter import CrewAIAdapter
from forge1.integrations.autogen_adapter import AutoGenAdapter

logger = logging.getLogger(__name__)

class EnhancedAgentFactory(AgentFactory):
    """Enhanced Agent Factory with Forge 1 capabilities"""
    
    # Extended agent classes mapping
    _enhanced_agent_classes: Dict[AgentType, Type[EnhancedBaseAgent]] = {
        **AgentFactory._agent_classes,  # Include Microsoft's base agents
        AgentType.PLANNER: SuperhumanPlannerAgent,
        AgentType.GROUP_CHAT_MANAGER: MultiModelCoordinator,
        # Add new Forge 1 specific agent types
        "COMPLIANCE": ComplianceAgent,
        "PERFORMANCE_OPTIMIZER": PerformanceOptimizerAgent,
    }
    
    # Framework adapters
    _framework_adapters = {
        "langchain": LangChainAdapter,
        "crewai": CrewAIAdapter,
        "autogen": AutoGenAdapter,
    }
    
    @classmethod
    async def create_enhanced_agent(
        cls,
        agent_type: AgentType,
        session_id: str,
        user_id: str,
        model_router: ModelRouter,
        temperature: float = 0.0,
        memory_store: Optional[CosmosMemoryContext] = None,
        system_message: Optional[str] = None,
        framework: Optional[str] = None,
        **kwargs,
    ) -> EnhancedBaseAgent:
        """Create an enhanced agent with Forge 1 capabilities
        
        Args:
            agent_type: The type of agent to create
            session_id: Session identifier
            user_id: User identifier
            model_router: Model router for multi-model capabilities
            temperature: Model temperature setting
            memory_store: Memory context store
            system_message: Custom system message
            framework: Optional framework to use (langchain, crewai, autogen)
            **kwargs: Additional parameters
            
        Returns:
            Enhanced agent instance with Forge 1 capabilities
        """
        
        # Check cache first
        cache_key = f"{session_id}_{agent_type}_{framework or 'default'}"
        if cache_key in cls._agent_cache:
            logger.info(f"Returning cached enhanced agent: {cache_key}")
            return cls._agent_cache[cache_key]
        
        # Get enhanced agent class
        agent_class = cls._enhanced_agent_classes.get(agent_type)
        if not agent_class:
            # Fallback to Microsoft's base agent with enhancements
            base_agent = await super().create_agent(
                agent_type=agent_type,
                session_id=session_id,
                user_id=user_id,
                temperature=temperature,
                memory_store=memory_store,
                system_message=system_message,
                **kwargs
            )
            # Wrap with enhancements
            agent_class = EnhancedBaseAgent
            enhanced_agent = await agent_class.enhance_existing_agent(
                base_agent=base_agent,
                model_router=model_router,
                **kwargs
            )
        else:
            # Create new enhanced agent
            if memory_store is None:
                memory_store = CosmosMemoryContext(session_id, user_id)
            
            # Apply framework adapter if specified
            if framework and framework in cls._framework_adapters:
                adapter_class = cls._framework_adapters[framework]
                adapter = adapter_class(model_router=model_router)
                enhanced_agent = await adapter.create_agent(
                    agent_type=agent_type,
                    agent_class=agent_class,
                    session_id=session_id,
                    user_id=user_id,
                    memory_store=memory_store,
                    system_message=system_message,
                    **kwargs
                )
            else:
                # Standard enhanced agent creation
                agent_init_params = inspect.signature(agent_class.__init__).parameters
                valid_keys = set(agent_init_params.keys()) - {"self"}
                filtered_kwargs = {
                    k: v for k, v in {
                        "agent_name": agent_type.value if hasattr(agent_type, 'value') else str(agent_type),
                        "session_id": session_id,
                        "user_id": user_id,
                        "memory_store": memory_store,
                        "system_message": system_message,
                        "model_router": model_router,
                        **kwargs,
                    }.items() if k in valid_keys
                }
                
                enhanced_agent = await agent_class.create(**filtered_kwargs)
        
        # Cache the enhanced agent
        cls._agent_cache[cache_key] = enhanced_agent
        
        logger.info(f"Created enhanced agent: {agent_type} with framework: {framework or 'default'}")
        return enhanced_agent
    
    @classmethod
    async def create_all_agents(
        cls,
        session_id: str,
        user_id: str,
        model_router: ModelRouter,
        temperature: float = 0.0,
        memory_store: Optional[CosmosMemoryContext] = None,
        client: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, EnhancedBaseAgent]:
        """Create all enhanced agents for a session
        
        Args:
            session_id: Session identifier
            user_id: User identifier
            model_router: Model router instance
            temperature: Model temperature
            memory_store: Memory context store
            client: AI client (optional, will use model router)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary of enhanced agent instances
        """
        
        if memory_store is None:
            memory_store = CosmosMemoryContext(session_id, user_id)
        
        agents = {}
        
        # Phase 1: Create base agents with enhancements
        base_agent_types = [
            AgentType.HR,
            AgentType.MARKETING,
            AgentType.PRODUCT,
            AgentType.PROCUREMENT,
            AgentType.TECH_SUPPORT,
            AgentType.GENERIC,
            AgentType.HUMAN,
        ]
        
        for agent_type in base_agent_types:
            enhanced_agent = await cls.create_enhanced_agent(
                agent_type=agent_type,
                session_id=session_id,
                user_id=user_id,
                model_router=model_router,
                temperature=temperature,
                memory_store=memory_store,
                **kwargs
            )
            agents[agent_type.value] = enhanced_agent
        
        # Phase 2: Create Forge 1 specific agents
        forge1_agents = {
            "COMPLIANCE": ComplianceAgent,
            "PERFORMANCE_OPTIMIZER": PerformanceOptimizerAgent,
        }
        
        for agent_name, agent_class in forge1_agents.items():
            enhanced_agent = await cls.create_enhanced_agent(
                agent_type=agent_name,
                session_id=session_id,
                user_id=user_id,
                model_router=model_router,
                temperature=temperature,
                memory_store=memory_store,
                **kwargs
            )
            agents[agent_name] = enhanced_agent
        
        # Phase 3: Create superhuman planner with all agent references
        planner_agent = await cls.create_enhanced_agent(
            agent_type=AgentType.PLANNER,
            session_id=session_id,
            user_id=user_id,
            model_router=model_router,
            temperature=temperature,
            memory_store=memory_store,
            agent_instances=agents,
            **kwargs
        )
        agents[AgentType.PLANNER.value] = planner_agent
        
        # Phase 4: Create multi-model coordinator as group chat manager
        coordinator = await cls.create_enhanced_agent(
            agent_type=AgentType.GROUP_CHAT_MANAGER,
            session_id=session_id,
            user_id=user_id,
            model_router=model_router,
            temperature=temperature,
            memory_store=memory_store,
            agent_instances=agents,
            **kwargs
        )
        agents[AgentType.GROUP_CHAT_MANAGER.value] = coordinator
        
        logger.info(f"Created {len(agents)} enhanced agents for session {session_id}")
        return agents
    
    @classmethod
    async def create_framework_integrated_agent(
        cls,
        framework: str,
        agent_config: Dict[str, Any],
        session_id: str,
        user_id: str,
        model_router: ModelRouter,
        **kwargs
    ) -> EnhancedBaseAgent:
        """Create an agent using a specific framework integration
        
        Args:
            framework: Framework name (langchain, crewai, autogen)
            agent_config: Framework-specific configuration
            session_id: Session identifier
            user_id: User identifier
            model_router: Model router instance
            **kwargs: Additional parameters
            
        Returns:
            Framework-integrated enhanced agent
        """
        
        if framework not in cls._framework_adapters:
            raise ValueError(f"Unsupported framework: {framework}")
        
        adapter_class = cls._framework_adapters[framework]
        adapter = adapter_class(model_router=model_router)
        
        return await adapter.create_framework_agent(
            config=agent_config,
            session_id=session_id,
            user_id=user_id,
            **kwargs
        )
    
    @classmethod
    def get_supported_frameworks(cls) -> List[str]:
        """Get list of supported framework integrations"""
        return list(cls._framework_adapters.keys())
    
    @classmethod
    async def optimize_agent_performance(
        cls,
        agent: EnhancedBaseAgent,
        performance_metrics: Dict[str, float]
    ) -> EnhancedBaseAgent:
        """Optimize agent performance based on metrics
        
        Args:
            agent: Agent to optimize
            performance_metrics: Current performance metrics
            
        Returns:
            Optimized agent instance
        """
        
        if hasattr(agent, 'optimize_performance'):
            return await agent.optimize_performance(performance_metrics)
        
        logger.warning(f"Agent {agent.agent_name} does not support performance optimization")
        return agent