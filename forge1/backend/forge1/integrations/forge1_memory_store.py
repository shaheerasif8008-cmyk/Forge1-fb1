"""
Forge1 Memory Store

MCAE memory store adapter that routes all memory operations through Forge1's
MemoryManager, ensuring tenant isolation, employee-specific namespacing,
and compliance with enterprise security policies.
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import sys
import os

# Add MCAE to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/src/backend'))

from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.memory_manager import MemoryManager
from forge1.core.memory_models import (
    MemoryContext, MemoryQuery, MemoryType, SecurityLevel, 
    MemorySearchResponse, MemorySearchResult
)

# Import MCAE memory context
from context.cosmos_memory_kernel import CosmosMemoryContext
from models.messages_kernel import (
    Plan, Step, AgentMessage, StepStatus, HumanFeedbackStatus,
    BaseDataModel
)

logger = logging.getLogger(__name__)


class Forge1MemoryStore(CosmosMemoryContext):
    """
    MCAE memory store that uses Forge1's memory manager.
    
    This adapter ensures all MCAE memory operations go through Forge1's
    enterprise memory management system with proper tenant isolation
    and employee-specific namespacing.
    """
    
    def __init__(self, session_id: str, user_id: str, tenant_id: str, 
                 employee_id: str, memory_manager: MemoryManager):
        """
        Initialize Forge1 memory store.
        
        Args:
            session_id: MCAE session ID
            user_id: User ID (same as employee_id in this context)
            tenant_id: Tenant ID for isolation
            employee_id: Employee ID for namespacing
            memory_manager: Forge1's memory manager instance
        """
        super().__init__(session_id, user_id)
        
        self.tenant_id = tenant_id
        self.employee_id = employee_id
        self.memory_manager = memory_manager
        
        # Memory namespace for this employee
        self.memory_namespace = f"tenant_{tenant_id}_employee_{employee_id}"
        
        # Cache for frequently accessed items
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Metrics
        self.metrics = {
            "items_stored": 0,
            "items_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
    async def initialize(self):
        """Initialize the memory store"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Initialize Forge1 memory manager if needed
            if not self.memory_manager._initialized:
                await self.memory_manager.initialize()
            
            logger.info(f"Initialized Forge1MemoryStore for employee {self.employee_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Forge1MemoryStore: {e}")
            raise
    
    async def add_item(self, item: BaseDataModel) -> str:
        """
        Store item through Forge1 memory manager with tenant isolation.
        
        Args:
            item: MCAE data model item to store
            
        Returns:
            Item ID
        """
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Convert MCAE item to Forge1 memory context
            memory_context = await self._convert_to_memory_context(item)
            
            # Store through Forge1 memory manager
            memory_id = await self.memory_manager.store_memory(memory_context)
            
            # Update cache
            cache_key = self._get_cache_key(item.data_type, getattr(item, 'id', memory_id))
            self._cache[cache_key] = {
                "item": item,
                "memory_id": memory_id,
                "timestamp": datetime.now(timezone.utc)
            }
            
            # Update metrics
            self.metrics["items_stored"] += 1
            
            logger.debug(f"Stored {item.data_type} item with ID {memory_id}")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store item: {e}")
            raise
    
    async def get_data_by_type(self, data_type: str) -> List[BaseDataModel]:
        """
        Retrieve data by type through Forge1 memory manager.
        
        Args:
            data_type: Type of data to retrieve
            
        Returns:
            List of items of the specified type
        """
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Create memory query for this data type
            query = MemoryQuery(
                tags=[f"data_type:{data_type}", f"employee:{self.employee_id}"],
                limit=100
            )
            
            # Search through Forge1 memory manager
            search_response = await self.memory_manager.search_memories(query, self.employee_id)
            
            # Convert results back to MCAE format
            items = []
            for result in search_response.results:
                item = await self._convert_from_memory_context(result.memory, data_type)
                if item:
                    items.append(item)
            
            # Update metrics
            self.metrics["items_retrieved"] += len(items)
            
            logger.debug(f"Retrieved {len(items)} items of type {data_type}")
            return items
            
        except Exception as e:
            logger.error(f"Failed to retrieve data by type {data_type}: {e}")
            return []
    
    async def get_data_by_type_and_session_id(self, data_type: str, session_id: str) -> List[BaseDataModel]:
        """Retrieve data by type and session ID"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Create memory query
            query = MemoryQuery(
                tags=[
                    f"data_type:{data_type}", 
                    f"session:{session_id}",
                    f"employee:{self.employee_id}"
                ],
                limit=100
            )
            
            # Search through Forge1 memory manager
            search_response = await self.memory_manager.search_memories(query, self.employee_id)
            
            # Convert results
            items = []
            for result in search_response.results:
                item = await self._convert_from_memory_context(result.memory, data_type)
                if item and getattr(item, 'session_id', None) == session_id:
                    items.append(item)
            
            logger.debug(f"Retrieved {len(items)} items of type {data_type} for session {session_id}")
            return items
            
        except Exception as e:
            logger.error(f"Failed to retrieve data by type and session: {e}")
            return []
    
    async def get_plan_by_session(self, session_id: str) -> Optional[Plan]:
        """Get plan by session ID"""
        plans = await self.get_data_by_type_and_session_id("plan", session_id)
        return plans[0] if plans else None
    
    async def get_plan_by_plan_id(self, plan_id: str) -> Optional[Plan]:
        """Get plan by plan ID"""
        try:
            # Check cache first
            cache_key = self._get_cache_key("plan", plan_id)
            if cache_key in self._cache:
                cached_data = self._cache[cache_key]
                if self._is_cache_valid(cached_data["timestamp"]):
                    self.metrics["cache_hits"] += 1
                    return cached_data["item"]
            
            self.metrics["cache_misses"] += 1
            
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Create memory query
            query = MemoryQuery(
                tags=[f"data_type:plan", f"plan_id:{plan_id}", f"employee:{self.employee_id}"],
                limit=1
            )
            
            # Search through Forge1 memory manager
            search_response = await self.memory_manager.search_memories(query, self.employee_id)
            
            if search_response.results:
                plan = await self._convert_from_memory_context(search_response.results[0].memory, "plan")
                
                # Cache the result
                self._cache[cache_key] = {
                    "item": plan,
                    "timestamp": datetime.now(timezone.utc)
                }
                
                return plan
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get plan by ID {plan_id}: {e}")
            return None
    
    async def get_steps_by_plan(self, plan_id: str) -> List[Step]:
        """Get steps for a plan"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Create memory query
            query = MemoryQuery(
                tags=[f"data_type:step", f"plan_id:{plan_id}", f"employee:{self.employee_id}"],
                limit=100
            )
            
            # Search through Forge1 memory manager
            search_response = await self.memory_manager.search_memories(query, self.employee_id)
            
            # Convert results
            steps = []
            for result in search_response.results:
                step = await self._convert_from_memory_context(result.memory, "step")
                if step and step.plan_id == plan_id:
                    steps.append(step)
            
            # Sort steps by creation order
            steps.sort(key=lambda s: getattr(s, 'ts', 0))
            
            logger.debug(f"Retrieved {len(steps)} steps for plan {plan_id}")
            return steps
            
        except Exception as e:
            logger.error(f"Failed to get steps for plan {plan_id}: {e}")
            return []
    
    async def get_steps_for_plan(self, plan_id: str) -> List[Step]:
        """Alias for get_steps_by_plan"""
        return await self.get_steps_by_plan(plan_id)
    
    async def get_all_plans(self) -> List[Plan]:
        """Get all plans for the current employee"""
        return await self.get_data_by_type("plan")
    
    async def update_step(self, step: Step) -> bool:
        """Update a step"""
        try:
            # Set tenant context
            set_current_tenant(self.tenant_id)
            
            # Find existing memory for this step
            query = MemoryQuery(
                tags=[f"data_type:step", f"step_id:{step.id}", f"employee:{self.employee_id}"],
                limit=1
            )
            
            search_response = await self.memory_manager.search_memories(query, self.employee_id)
            
            if search_response.results:
                memory_id = search_response.results[0].memory.id
                
                # Update the memory with new step data
                updates = {
                    "content": self._serialize_item(step),
                    "summary": f"Step {step.id}: {step.action}",
                    "tags": self._get_tags_for_item(step)
                }
                
                success = await self.memory_manager.update_memory(memory_id, updates, self.employee_id)
                
                if success:
                    # Update cache
                    cache_key = self._get_cache_key("step", step.id)
                    self._cache[cache_key] = {
                        "item": step,
                        "timestamp": datetime.now(timezone.utc)
                    }
                
                return success
            else:
                # Step doesn't exist, create it
                await self.add_item(step)
                return True
                
        except Exception as e:
            logger.error(f"Failed to update step {step.id}: {e}")
            return False
    
    async def _convert_to_memory_context(self, item: BaseDataModel) -> MemoryContext:
        """Convert MCAE item to Forge1 MemoryContext"""
        
        # Serialize the item
        content = self._serialize_item(item)
        
        # Generate summary
        summary = self._generate_summary(item)
        
        # Extract keywords
        keywords = self._extract_keywords(item)
        
        # Get tags
        tags = self._get_tags_for_item(item)
        
        # Create memory context
        memory_context = MemoryContext(
            id=str(uuid.uuid4()),
            employee_id=self.employee_id,
            session_id=self.session_id,
            memory_type=self._get_memory_type(item.data_type),
            content=content,
            summary=summary,
            keywords=keywords,
            security_level=SecurityLevel.PRIVATE,  # Employee-specific data
            owner_id=self.employee_id,
            source=f"mcae_{item.data_type}",
            tags=tags,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        
        return memory_context
    
    async def _convert_from_memory_context(self, memory: MemoryContext, data_type: str) -> Optional[BaseDataModel]:
        """Convert Forge1 MemoryContext back to MCAE item"""
        try:
            # Deserialize the content
            if isinstance(memory.content, dict):
                item_data = memory.content
            else:
                item_data = json.loads(memory.content)
            
            # Create appropriate MCAE object based on data type
            if data_type == "plan":
                return Plan(**item_data)
            elif data_type == "step":
                return Step(**item_data)
            elif data_type == "agent_message":
                return AgentMessage(**item_data)
            else:
                # Generic handling for unknown types
                return BaseDataModel(**item_data)
                
        except Exception as e:
            logger.error(f"Failed to convert memory context to {data_type}: {e}")
            return None
    
    def _serialize_item(self, item: BaseDataModel) -> Dict:
        """Serialize MCAE item to dictionary"""
        if hasattr(item, 'model_dump'):
            return item.model_dump()
        elif hasattr(item, 'dict'):
            return item.dict()
        else:
            return item.__dict__
    
    def _generate_summary(self, item: BaseDataModel) -> str:
        """Generate summary for the item"""
        if hasattr(item, 'summary') and item.summary:
            return item.summary
        elif hasattr(item, 'action') and item.action:
            return f"{item.data_type}: {item.action}"
        elif hasattr(item, 'content') and item.content:
            content_str = str(item.content)
            return f"{item.data_type}: {content_str[:100]}..."
        else:
            return f"{item.data_type} item"
    
    def _extract_keywords(self, item: BaseDataModel) -> List[str]:
        """Extract keywords from the item"""
        keywords = [item.data_type]
        
        # Add item-specific keywords
        if hasattr(item, 'agent') and item.agent:
            agent_str = item.agent.value if hasattr(item.agent, 'value') else str(item.agent)
            keywords.append(agent_str)
        
        if hasattr(item, 'status') and item.status:
            status_str = item.status.value if hasattr(item.status, 'value') else str(item.status)
            keywords.append(status_str)
        
        return keywords
    
    def _get_tags_for_item(self, item: BaseDataModel) -> List[str]:
        """Get tags for the item"""
        tags = [
            f"tenant:{self.tenant_id}",
            f"employee:{self.employee_id}",
            f"session:{self.session_id}",
            f"data_type:{item.data_type}"
        ]
        
        # Add item-specific tags
        if hasattr(item, 'id') and item.id:
            tags.append(f"{item.data_type}_id:{item.id}")
        
        if hasattr(item, 'plan_id') and item.plan_id:
            tags.append(f"plan_id:{item.plan_id}")
        
        return tags
    
    def _get_memory_type(self, data_type: str) -> MemoryType:
        """Map MCAE data type to Forge1 memory type"""
        mapping = {
            "plan": MemoryType.WORKFLOW,
            "step": MemoryType.TASK,
            "agent_message": MemoryType.CONVERSATION,
            "human_feedback": MemoryType.FEEDBACK
        }
        
        return mapping.get(data_type, MemoryType.GENERAL)
    
    def _get_cache_key(self, data_type: str, item_id: str) -> str:
        """Generate cache key"""
        return f"{self.tenant_id}_{self.employee_id}_{data_type}_{item_id}"
    
    def _is_cache_valid(self, timestamp: datetime) -> bool:
        """Check if cache entry is still valid"""
        age = (datetime.now(timezone.utc) - timestamp).total_seconds()
        return age < self._cache_ttl
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory store metrics"""
        return {
            **self.metrics,
            "cache_size": len(self._cache),
            "tenant_id": self.tenant_id,
            "employee_id": self.employee_id,
            "session_id": self.session_id
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.info(f"Cleared cache for employee {self.employee_id}")
    
    async def cleanup(self):
        """Clean up resources"""
        self.clear_cache()
        logger.info(f"Cleaned up Forge1MemoryStore for employee {self.employee_id}")