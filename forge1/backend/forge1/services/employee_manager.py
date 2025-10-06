# forge1/backend/forge1/services/employee_manager.py
"""
Employee Manager Core Service

Central service for managing AI employee lifecycle, configuration, and interactions.
Coordinates between client management, memory management, and AI processing.

Requirements: 1.2, 1.4, 3.1, 3.2, 4.4
"""

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal

from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.core.model_router import ModelRouter
from forge1.services.client_manager import ClientManager
from forge1.services.employee_memory_manager import EmployeeMemoryManager
from forge1.models.employee_models import (
    Employee, EmployeeRequirements, EmployeeInteraction, EmployeeResponse,
    EmployeeStatus, MemoryType, PersonalityConfig, ModelPreferences,
    CommunicationStyle, FormalityLevel, ExpertiseLevel, ResponseLength,
    EmployeeNotFoundError, ClientNotFoundError, InvalidConfigurationError,
    TenantIsolationError, generate_employee_id, MemoryItem
)

# MCAE Integration imports
from forge1.integrations.mcae_adapter import MCAEAdapter, MCAEIntegrationError, WorkflowExecutionError

logger = logging.getLogger(__name__)


class EmployeeManager:
    """
    Central service for AI employee lifecycle management.
    
    Features:
    - Employee creation from requirements
    - Employee loading and caching
    - Configuration management
    - Status tracking
    - Integration with memory and client systems
    """
    
    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        client_manager: Optional[ClientManager] = None,
        memory_manager: Optional[EmployeeMemoryManager] = None,
        model_router: Optional[ModelRouter] = None,
        mcae_adapter: Optional[MCAEAdapter] = None
    ):
        self.db_manager = db_manager
        self.client_manager = client_manager
        self.memory_manager = memory_manager
        self.model_router = model_router
        self.mcae_adapter = mcae_adapter
        self._initialized = False
        
        # Employee cache for performance
        self._employee_cache = {}
        self._cache_ttl = 3600  # 1 hour cache TTL
        
        # Performance metrics
        self._metrics = {
            "employees_created": 0,
            "employees_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "interactions_processed": 0,
            "average_load_time": 0.0,
            "mcae_workflows_created": 0,
            "mcae_tasks_executed": 0
        }
    
    async def initialize(self):
        """Initialize the employee manager and dependencies"""
        if self._initialized:
            return
        
        # Initialize database manager
        if not self.db_manager:
            self.db_manager = await get_database_manager()
        
        # Initialize client manager
        if not self.client_manager:
            self.client_manager = ClientManager(self.db_manager)
            await self.client_manager.initialize()
        
        # Initialize memory manager
        if not self.memory_manager:
            self.memory_manager = EmployeeMemoryManager(self.db_manager)
            await self.memory_manager.initialize()
        
        # Initialize model router
        if not self.model_router:
            self.model_router = ModelRouter()
        
        # Initialize MCAE adapter if not provided
        if not self.mcae_adapter:
            self.mcae_adapter = MCAEAdapter(
                employee_manager=self,
                model_router=self.model_router,
                memory_manager=self.memory_manager
            )
            await self.mcae_adapter.initialize()
        
        self._initialized = True
        logger.info("Employee Manager initialized with all dependencies including MCAE")
    
    async def create_employee(
        self,
        client_id: str,
        requirements: EmployeeRequirements
    ) -> Employee:
        """
        Create a new AI employee from client requirements.
        
        This is the main entry point for employee creation, coordinating
        with client management and memory initialization.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Use client manager to create employee
            employee = await self.client_manager.create_employee_for_client(
                client_id, requirements
            )
            
            # Initialize employee memory namespace
            await self.memory_manager.initialize_employee_namespace(
                client_id, employee.id
            )
            
            # NEW: Register employee in MCAE
            try:
                if self.mcae_adapter:
                    workflow_id = await self.mcae_adapter.register_employee_workflow(employee)
                    employee.workflow_id = workflow_id
                    await self._update_employee_workflow_id(employee.id, workflow_id)
                    self._metrics["mcae_workflows_created"] += 1
                    logger.info(f"Registered employee {employee.id} with MCAE workflow {workflow_id}")
            except MCAEIntegrationError as e:
                logger.warning(f"Failed to register employee {employee.id} with MCAE: {e}")
                # Continue without MCAE integration - graceful degradation
            
            # Cache the new employee
            cache_key = self._get_cache_key(client_id, employee.id)
            self._employee_cache[cache_key] = {
                "employee": employee,
                "cached_at": time.time()
            }
            
            # Update metrics
            self._metrics["employees_created"] += 1
            
            creation_time = time.time() - start_time
            logger.info(f"Created employee {employee.id} for client {client_id} in {creation_time:.3f}s")
            
            return employee
            
        except Exception as e:
            logger.error(f"Failed to create employee for client {client_id}: {e}")
            raise
    
    async def load_employee(
        self,
        client_id: str,
        employee_id: str,
        include_memory_context: bool = True,
        memory_limit: int = 10
    ) -> Optional[Employee]:
        """
        Load employee configuration and context for interaction.
        
        This is called before each interaction to get the employee's
        current state, preferences, and relevant memory context.
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Check cache first
            cache_key = self._get_cache_key(client_id, employee_id)
            cached_data = self._employee_cache.get(cache_key)
            
            if cached_data and (time.time() - cached_data["cached_at"]) < self._cache_ttl:
                employee = cached_data["employee"]
                self._metrics["cache_hits"] += 1
                logger.debug(f"Loaded employee {employee_id} from cache")
            else:
                # Load from database
                employee = await self._load_employee_from_database(client_id, employee_id)
                if not employee:
                    return None
                
                # Cache the employee
                self._employee_cache[cache_key] = {
                    "employee": employee,
                    "cached_at": time.time()
                }
                self._metrics["cache_misses"] += 1
            
            # Load memory context if requested
            if include_memory_context:
                try:
                    memory_context = await self.memory_manager.get_employee_context(
                        client_id, employee_id, limit=memory_limit
                    )
                    employee.memory_context = memory_context
                except Exception as e:
                    logger.warning(f"Failed to load memory context for {employee_id}: {e}")
                    employee.memory_context = []
            
            # Update metrics
            self._metrics["employees_loaded"] += 1
            load_time = time.time() - start_time
            
            # Update average load time
            current_avg = self._metrics["average_load_time"]
            load_count = self._metrics["employees_loaded"]
            self._metrics["average_load_time"] = (
                (current_avg * (load_count - 1) + load_time) / load_count
            )
            
            logger.debug(f"Loaded employee {employee_id} in {load_time:.3f}s")
            
            return employee
            
        except Exception as e:
            logger.error(f"Failed to load employee {employee_id}: {e}")
            raise EmployeeNotFoundError(f"Failed to load employee: {e}")
    
    async def get_employee(
        self,
        client_id: str,
        employee_id: str
    ) -> Optional[Employee]:
        """
        Get employee basic information without memory context.
        Lighter weight version of load_employee for simple queries.
        """
        return await self.load_employee(
            client_id, employee_id, 
            include_memory_context=False
        )
    
    async def update_employee(
        self,
        client_id: str,
        employee_id: str,
        updates: Dict[str, Any]
    ) -> Employee:
        """
        Update employee configuration.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Prepare update fields for database
            update_fields = []
            update_params = []
            param_count = 0
            
            # Handle personality updates
            if 'personality' in updates:
                personality_data = updates['personality']
                
                # Update individual personality fields
                personality_fields = {
                    'communication_style': 'communication_style',
                    'formality_level': 'formality_level',
                    'expertise_level': 'expertise_level',
                    'response_length': 'response_length',
                    'creativity_level': 'creativity_level',
                    'empathy_level': 'empathy_level'
                }
                
                for field, db_field in personality_fields.items():
                    if field in personality_data:
                        param_count += 1
                        update_fields.append(f"{db_field} = ${param_count}")
                        
                        value = personality_data[field]
                        if hasattr(value, 'value'):
                            value = value.value
                        update_params.append(value)
                
                # Update custom traits
                if 'custom_traits' in personality_data:
                    param_count += 1
                    update_fields.append(f"personality = ${param_count}")
                    update_params.append(personality_data['custom_traits'])
            
            # Handle model preferences updates
            if 'model_preferences' in updates:
                model_data = updates['model_preferences']
                
                model_fields = {
                    'primary_model': 'primary_model',
                    'fallback_models': 'fallback_models',
                    'temperature': 'temperature',
                    'max_tokens': 'max_tokens',
                    'specialized_models': 'specialized_models'
                }
                
                for field, db_field in model_fields.items():
                    if field in model_data:
                        param_count += 1
                        update_fields.append(f"{db_field} = ${param_count}")
                        update_params.append(model_data[field])
            
            # Handle other direct updates
            direct_fields = {
                'name': 'name',
                'role': 'role',
                'tool_access': 'tool_access',
                'knowledge_sources': 'knowledge_sources',
                'status': 'status'
            }
            
            for field, db_field in direct_fields.items():
                if field in updates:
                    param_count += 1
                    update_fields.append(f"{db_field} = ${param_count}")
                    
                    value = updates[field]
                    if hasattr(value, 'value'):
                        value = value.value
                    update_params.append(value)
            
            if not update_fields:
                return employee  # No updates to apply
            
            # Add updated_at timestamp
            param_count += 1
            update_fields.append(f"updated_at = ${param_count}")
            update_params.append(datetime.now(timezone.utc))
            
            # Add WHERE clause parameters
            param_count += 1
            update_params.append(employee_id)
            param_count += 1
            update_params.append(client_id)
            
            # Execute update
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute(f"""
                    UPDATE forge1_employees.employees
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count - 1} AND client_id = ${param_count}
                """, *update_params)
            
            # Clear cache to force reload
            cache_key = self._get_cache_key(client_id, employee_id)
            if cache_key in self._employee_cache:
                del self._employee_cache[cache_key]
            
            # Return updated employee
            updated_employee = await self.get_employee(client_id, employee_id)
            
            logger.info(f"Updated employee {employee_id} configuration")
            return updated_employee
            
        except Exception as e:
            logger.error(f"Failed to update employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to update employee: {e}")
    
    async def list_employees(
        self,
        client_id: str,
        status: Optional[EmployeeStatus] = None,
        role: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Employee]:
        """
        List employees for a client with optional filtering.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Use client manager to get employees
            employees = await self.client_manager.get_client_employees(
                client_id, status, limit, offset
            )
            
            # Filter by role if specified
            if role:
                employees = [emp for emp in employees if role.lower() in emp.role.lower()]
            
            return employees
            
        except Exception as e:
            logger.error(f"Failed to list employees for client {client_id}: {e}")
            raise ClientNotFoundError(f"Failed to list employees: {e}")
    
    async def delete_employee(
        self,
        client_id: str,
        employee_id: str,
        archive_memory: bool = True
    ) -> bool:
        """
        Delete (archive) an employee and optionally preserve memory.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get employee to verify it exists
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            if archive_memory:
                # Archive employee (soft delete)
                await self.update_employee(
                    client_id, employee_id, 
                    {"status": EmployeeStatus.ARCHIVED}
                )
                logger.info(f"Archived employee {employee_id}")
            else:
                # Hard delete from database
                async with self.db_manager.postgres.acquire() as conn:
                    # Set RLS context
                    await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                    
                    # Delete employee (cascades to interactions, summaries, knowledge)
                    await conn.execute("""
                        DELETE FROM forge1_employees.employees
                        WHERE id = $1 AND client_id = $2
                    """, employee_id, client_id)
                
                logger.info(f"Deleted employee {employee_id}")
            
            # Clear from cache
            cache_key = self._get_cache_key(client_id, employee_id)
            if cache_key in self._employee_cache:
                del self._employee_cache[cache_key]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to delete employee: {e}")
    
    async def get_employee_stats(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """
        Get comprehensive statistics for an employee.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get employee info
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Get memory statistics
            memory_stats = await self.memory_manager.get_memory_stats(client_id, employee_id)
            
            # Get employee-specific stats from database
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Get interaction performance stats
                perf_stats = await conn.fetchrow("""
                    SELECT 
                        AVG(processing_time_ms) as avg_processing_time,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost,
                        COUNT(DISTINCT DATE(timestamp)) as active_days
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1 AND employee_id = $2
                """, client_id, employee_id)
            
            # Combine all statistics
            stats = {
                "employee_id": employee_id,
                "employee_name": employee.name,
                "employee_role": employee.role,
                "status": employee.status.value,
                "created_at": employee.created_at.isoformat(),
                "last_interaction_at": employee.last_interaction_at.isoformat() if employee.last_interaction_at else None,
                
                # Memory statistics
                **memory_stats,
                
                # Performance statistics
                "average_processing_time_ms": float(perf_stats['avg_processing_time'] or 0.0),
                "total_tokens_used": int(perf_stats['total_tokens'] or 0),
                "total_cost": float(perf_stats['total_cost'] or Decimal('0.00')),
                "active_days": perf_stats['active_days'] or 0,
                
                # Configuration
                "personality": {
                    "communication_style": employee.personality.communication_style.value,
                    "formality_level": employee.personality.formality_level.value,
                    "expertise_level": employee.personality.expertise_level.value,
                    "creativity_level": employee.personality.creativity_level,
                    "empathy_level": employee.personality.empathy_level
                },
                "model_preferences": {
                    "primary_model": employee.model_preferences.primary_model,
                    "temperature": employee.model_preferences.temperature,
                    "max_tokens": employee.model_preferences.max_tokens
                },
                "tools_count": len(employee.tool_access),
                "knowledge_sources_count": len(employee.knowledge_sources)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stats for employee {employee_id}: {e}")
            raise EmployeeNotFoundError(f"Failed to get employee statistics: {e}")
    
    async def search_employees(
        self,
        client_id: str,
        query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search employees by name, role, or capabilities.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Search in database
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Full-text search across name, role, and tools
                rows = await conn.fetch("""
                    SELECT 
                        id, name, role, communication_style, expertise_level,
                        tool_access, knowledge_sources, status, created_at,
                        last_interaction_at,
                        ts_rank(
                            to_tsvector('english', name || ' ' || role || ' ' || array_to_string(tool_access, ' ')),
                            plainto_tsquery('english', $2)
                        ) as rank
                    FROM forge1_employees.employees
                    WHERE client_id = $1
                    AND (
                        to_tsvector('english', name || ' ' || role || ' ' || array_to_string(tool_access, ' '))
                        @@ plainto_tsquery('english', $2)
                        OR name ILIKE $3
                        OR role ILIKE $3
                    )
                    ORDER BY rank DESC, name ASC
                    LIMIT $4
                """, client_id, query, f"%{query}%", limit)
            
            # Format results
            results = []
            for row in rows:
                results.append({
                    "id": row['id'],
                    "name": row['name'],
                    "role": row['role'],
                    "communication_style": row['communication_style'],
                    "expertise_level": row['expertise_level'],
                    "tools": row['tool_access'],
                    "knowledge_sources": row['knowledge_sources'],
                    "status": row['status'],
                    "created_at": row['created_at'].isoformat(),
                    "last_interaction_at": row['last_interaction_at'].isoformat() if row['last_interaction_at'] else None,
                    "relevance_score": float(row['rank']) if row['rank'] else 0.0
                })
            
            logger.debug(f"Employee search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Employee search failed for client {client_id}: {e}")
            raise InvalidConfigurationError(f"Employee search failed: {e}")
    
    async def interact_with_employee(
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
        Handle interaction with an AI employee using the interaction processor.
        
        This is the main entry point for employee interactions from the API layer.
        """
        if not self._initialized:
            await self.initialize()
        
        # Import here to avoid circular imports
        from forge1.services.interaction_processor import InteractionProcessor
        
        # Create interaction processor if not already available
        if not hasattr(self, '_interaction_processor'):
            self._interaction_processor = InteractionProcessor(
                employee_manager=self,
                memory_manager=self.memory_manager,
                model_router=self.model_router,
                db_manager=self.db_manager
            )
            await self._interaction_processor.initialize()
        
        # Process the interaction
        response = await self._interaction_processor.process_interaction(
            client_id=client_id,
            employee_id=employee_id,
            message=message,
            session_id=session_id,
            context=context,
            include_memory=include_memory,
            memory_limit=memory_limit
        )
        
        # Update metrics
        self._metrics["interactions_processed"] += 1
        
        return response
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get employee manager performance metrics"""
        metrics = {
            **self._metrics,
            "cache_size": len(self._employee_cache),
            "cache_hit_rate": (
                self._metrics["cache_hits"] / 
                max(self._metrics["cache_hits"] + self._metrics["cache_misses"], 1)
            )
        }
        
        # Add interaction processor metrics if available
        if hasattr(self, '_interaction_processor'):
            interaction_metrics = self._interaction_processor.get_metrics()
            metrics["interaction_processor"] = interaction_metrics
        
        return metrics
    
    def clear_cache(self, client_id: Optional[str] = None, employee_id: Optional[str] = None):
        """Clear employee cache"""
        if client_id and employee_id:
            # Clear specific employee
            cache_key = self._get_cache_key(client_id, employee_id)
            if cache_key in self._employee_cache:
                del self._employee_cache[cache_key]
        elif client_id:
            # Clear all employees for client
            keys_to_remove = [
                key for key in self._employee_cache.keys() 
                if key.startswith(f"{client_id}:")
            ]
            for key in keys_to_remove:
                del self._employee_cache[key]
        else:
            # Clear all cache
            self._employee_cache.clear()
    
    # Private helper methods
    
    async def _validate_tenant_access(self, client_id: str) -> None:
        """Validate that current tenant can access the client"""
        current_tenant = get_current_tenant()
        if current_tenant and current_tenant != client_id:
            raise TenantIsolationError(
                f"Tenant {current_tenant} cannot access client {client_id}"
            )
    
    def _get_cache_key(self, client_id: str, employee_id: str) -> str:
        """Generate cache key for employee"""
        return f"{client_id}:{employee_id}"
    
    async def _load_employee_from_database(
        self,
        client_id: str,
        employee_id: str
    ) -> Optional[Employee]:
        """Load employee from database"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                row = await conn.fetchrow("""
                    SELECT 
                        id, client_id, name, role, personality, model_preferences,
                        tool_access, knowledge_sources, created_at, updated_at,
                        last_interaction_at, status, communication_style,
                        formality_level, expertise_level, response_length,
                        creativity_level, empathy_level, primary_model,
                        fallback_models, temperature, max_tokens, specialized_models
                    FROM forge1_employees.employees
                    WHERE id = $1 AND client_id = $2
                """, employee_id, client_id)
            
            if not row:
                return None
            
            # Reconstruct personality config
            personality = PersonalityConfig(
                communication_style=CommunicationStyle(row['communication_style']),
                formality_level=FormalityLevel(row['formality_level']),
                expertise_level=ExpertiseLevel(row['expertise_level']),
                response_length=ResponseLength(row['response_length']),
                creativity_level=row['creativity_level'],
                empathy_level=row['empathy_level'],
                custom_traits=row['personality'] or {}
            )
            
            # Reconstruct model preferences
            model_preferences = ModelPreferences(
                primary_model=row['primary_model'],
                fallback_models=row['fallback_models'],
                temperature=row['temperature'],
                max_tokens=row['max_tokens'],
                specialized_models=row['specialized_models'] or {}
            )
            
            # Create employee object
            employee = Employee(
                id=row['id'],
                client_id=row['client_id'],
                name=row['name'],
                role=row['role'],
                personality=personality,
                model_preferences=model_preferences,
                tool_access=row['tool_access'],
                knowledge_sources=row['knowledge_sources'],
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                last_interaction_at=row['last_interaction_at'],
                status=EmployeeStatus(row['status'])
            )
            
            return employee
            
        except Exception as e:
            logger.error(f"Failed to load employee {employee_id} from database: {e}")
            return None  
  # Configuration Management Methods for Task 11

    async def update_employee_personality(
        self,
        client_id: str,
        employee_id: str,
        personality_updates: Dict[str, Any]
    ) -> Employee:
        """Update employee personality traits."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Update personality fields
            updates = {"personality": personality_updates}
            return await self.update_employee(client_id, employee_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to update personality for employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to update personality: {e}")

    async def update_employee_model_preferences(
        self,
        client_id: str,
        employee_id: str,
        model_updates: Dict[str, Any]
    ) -> Employee:
        """Update employee model preferences."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Update model preferences
            updates = {"model_preferences": model_updates}
            return await self.update_employee(client_id, employee_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to update model preferences for employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to update model preferences: {e}")

    async def update_employee_tool_access(
        self,
        client_id: str,
        employee_id: str,
        tools: List[str],
        tool_credentials: Optional[Dict[str, Any]] = None
    ) -> Employee:
        """Update employee tool access and credentials."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Update tool access
            updates = {"tool_access": tools}
            
            # Store tool credentials securely if provided
            if tool_credentials:
                # In production, encrypt credentials before storing
                async with self.db_manager.postgres.acquire() as conn:
                    await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                    
                    # Store or update tool credentials
                    await conn.execute("""
                        INSERT INTO forge1_employees.employee_tool_credentials 
                        (client_id, employee_id, tool_credentials, updated_at)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (client_id, employee_id)
                        DO UPDATE SET 
                            tool_credentials = $3,
                            updated_at = $4
                    """, client_id, employee_id, tool_credentials, datetime.now(timezone.utc))
            
            return await self.update_employee(client_id, employee_id, updates)
            
        except Exception as e:
            logger.error(f"Failed to update tool access for employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to update tool access: {e}")

    async def get_employee_tool_usage_stats(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """Get tool usage statistics for an employee."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            async with self.db_manager.postgres.acquire() as conn:
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Get tool usage from interaction context
                tool_usage = await conn.fetch("""
                    SELECT 
                        context->>'tool_used' as tool_name,
                        COUNT(*) as usage_count,
                        AVG(processing_time_ms) as avg_processing_time,
                        MAX(timestamp) as last_used
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1 AND employee_id = $2
                    AND context->>'tool_used' IS NOT NULL
                    GROUP BY context->>'tool_used'
                    ORDER BY usage_count DESC
                """, client_id, employee_id)
            
            return {
                "tool_usage": [
                    {
                        "tool_name": row["tool_name"],
                        "usage_count": row["usage_count"],
                        "avg_processing_time_ms": float(row["avg_processing_time"] or 0),
                        "last_used": row["last_used"].isoformat() if row["last_used"] else None
                    }
                    for row in tool_usage
                ],
                "total_tool_interactions": sum(row["usage_count"] for row in tool_usage)
            }
            
        except Exception as e:
            logger.error(f"Failed to get tool usage stats for employee {employee_id}: {e}")
            return {"tool_usage": [], "total_tool_interactions": 0}

    async def add_employee_knowledge_source(
        self,
        client_id: str,
        employee_id: str,
        knowledge_id: str,
        title: str
    ) -> bool:
        """Add a knowledge source to employee's knowledge sources list."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Add knowledge source to list if not already present
            knowledge_sources = employee.knowledge_sources.copy()
            if knowledge_id not in knowledge_sources:
                knowledge_sources.append(knowledge_id)
                
                # Update employee
                await self.update_employee(
                    client_id, employee_id, 
                    {"knowledge_sources": knowledge_sources}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add knowledge source to employee {employee_id}: {e}")
            return False

    async def remove_employee_knowledge_source(
        self,
        client_id: str,
        employee_id: str,
        knowledge_id: str
    ) -> bool:
        """Remove a knowledge source from employee's knowledge sources list."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Remove knowledge source from list
            knowledge_sources = employee.knowledge_sources.copy()
            if knowledge_id in knowledge_sources:
                knowledge_sources.remove(knowledge_id)
                
                # Update employee
                await self.update_employee(
                    client_id, employee_id, 
                    {"knowledge_sources": knowledge_sources}
                )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove knowledge source from employee {employee_id}: {e}")
            return False

    async def reset_employee_configuration(
        self,
        client_id: str,
        employee_id: str,
        reset_type: str,
        default_values: Optional[Dict[str, Any]] = None
    ) -> Employee:
        """Reset employee configuration to defaults."""
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Get current employee
            employee = await self.get_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            updates = {}
            
            if reset_type in ["personality", "all"]:
                # Reset personality to defaults
                default_personality = {
                    "communication_style": "professional",
                    "formality_level": "formal",
                    "expertise_level": "intermediate",
                    "response_length": "detailed",
                    "creativity_level": 0.7,
                    "empathy_level": 0.8,
                    "custom_traits": {}
                }
                
                if default_values and "personality" in default_values:
                    default_personality.update(default_values["personality"])
                
                updates["personality"] = default_personality
            
            if reset_type in ["model_preferences", "all"]:
                # Reset model preferences to defaults
                default_models = {
                    "primary_model": "gpt-4",
                    "fallback_models": ["gpt-3.5-turbo"],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "specialized_models": {}
                }
                
                if default_values and "model_preferences" in default_values:
                    default_models.update(default_values["model_preferences"])
                
                updates["model_preferences"] = default_models
            
            if reset_type in ["tools", "all"]:
                # Reset tools to basic set
                default_tools = ["web_search", "calculator"]
                
                if default_values and "tools" in default_values:
                    default_tools = default_values["tools"]
                
                updates["tool_access"] = default_tools
            
            if reset_type in ["knowledge_sources", "all"]:
                # Clear knowledge sources
                updates["knowledge_sources"] = []
                
                # Also remove from knowledge base
                await self.memory_manager.clear_employee_knowledge_base(client_id, employee_id)
            
            # Apply updates
            if updates:
                return await self.update_employee(client_id, employee_id, updates)
            else:
                return employee
            
        except Exception as e:
            logger.error(f"Failed to reset configuration for employee {employee_id}: {e}")
            raise InvalidConfigurationError(f"Failed to reset configuration: {e}") 
   # MCAE Integration Methods
    
    async def execute_employee_task(
        self,
        client_id: str,
        employee_id: str,
        task: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Execute task through MCAE orchestration.
        
        Args:
            client_id: Client ID for tenant isolation
            employee_id: Employee ID to execute task with
            task: Task description to execute
            context: Additional context for the task
            
        Returns:
            Dictionary containing workflow execution results
            
        Raises:
            EmployeeNotFoundError: If employee not found
            WorkflowExecutionError: If MCAE execution fails
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Load employee
            employee = await self.load_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            # Check if employee has MCAE workflow
            if not employee.workflow_id:
                # Try to register employee with MCAE if not already registered
                if self.mcae_adapter:
                    try:
                        workflow_id = await self.mcae_adapter.register_employee_workflow(employee)
                        employee.workflow_id = workflow_id
                        await self._update_employee_workflow_id(employee.id, workflow_id)
                        logger.info(f"Registered employee {employee_id} with MCAE workflow {workflow_id}")
                    except MCAEIntegrationError as e:
                        logger.error(f"Failed to register employee {employee_id} with MCAE: {e}")
                        raise WorkflowExecutionError(f"Employee not registered with MCAE: {e}")
                else:
                    raise WorkflowExecutionError("MCAE adapter not available")
            
            # Execute workflow through MCAE
            result = await self.mcae_adapter.execute_workflow(
                employee.workflow_id, task, context or {}
            )
            
            # Update metrics
            self._metrics["mcae_tasks_executed"] += 1
            
            logger.info(f"Executed task for employee {employee_id} through MCAE")
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute task for employee {employee_id}: {e}")
            raise
    
    async def get_employee_workflow_status(
        self,
        client_id: str,
        employee_id: str
    ) -> Dict[str, Any]:
        """
        Get the current status of an employee's MCAE workflow.
        
        Args:
            client_id: Client ID for tenant isolation
            employee_id: Employee ID to check
            
        Returns:
            Dictionary containing workflow status information
        """
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Load employee
            employee = await self.load_employee(client_id, employee_id)
            if not employee:
                raise EmployeeNotFoundError(f"Employee {employee_id} not found")
            
            if not employee.workflow_id:
                return {
                    "employee_id": employee_id,
                    "workflow_registered": False,
                    "status": "not_registered"
                }
            
            # Get workflow status from MCAE
            if self.mcae_adapter:
                workflow_status = await self.mcae_adapter.get_workflow_status(employee.workflow_id)
                workflow_status["workflow_registered"] = True
                return workflow_status
            else:
                return {
                    "employee_id": employee_id,
                    "workflow_id": employee.workflow_id,
                    "workflow_registered": True,
                    "status": "mcae_unavailable"
                }
                
        except Exception as e:
            logger.error(f"Failed to get workflow status for employee {employee_id}: {e}")
            return {
                "employee_id": employee_id,
                "error": str(e),
                "status": "error"
            }
    
    async def cleanup_employee_workflow(
        self,
        client_id: str,
        employee_id: str
    ) -> bool:
        """
        Clean up an employee's MCAE workflow resources.
        
        Args:
            client_id: Client ID for tenant isolation
            employee_id: Employee ID to cleanup
            
        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            # Validate tenant access
            await self._validate_tenant_access(client_id)
            
            # Load employee
            employee = await self.load_employee(client_id, employee_id)
            if not employee or not employee.workflow_id:
                return True  # Nothing to cleanup
            
            # Cleanup workflow in MCAE
            if self.mcae_adapter:
                success = await self.mcae_adapter.cleanup_workflow(employee.workflow_id)
                if success:
                    # Clear workflow ID from employee record
                    await self._update_employee_workflow_id(employee.id, None)
                    logger.info(f"Cleaned up MCAE workflow for employee {employee_id}")
                return success
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup workflow for employee {employee_id}: {e}")
            return False
    
    async def _update_employee_workflow_id(
        self,
        employee_id: str,
        workflow_id: Optional[str]
    ) -> bool:
        """
        Update employee's workflow ID in the database.
        
        Args:
            employee_id: Employee ID to update
            workflow_id: New workflow ID (or None to clear)
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Update workflow_id field
                result = await conn.execute("""
                    UPDATE forge1_employees.employees
                    SET workflow_id = $1, updated_at = $2
                    WHERE id = $3
                """, workflow_id, datetime.now(timezone.utc), employee_id)
                
                return result == "UPDATE 1"
                
        except Exception as e:
            logger.error(f"Failed to update workflow ID for employee {employee_id}: {e}")
            return False
    
    def get_mcae_metrics(self) -> Dict[str, Any]:
        """Get MCAE-specific metrics"""
        base_metrics = self.get_metrics()
        
        mcae_metrics = {
            "mcae_workflows_created": self._metrics.get("mcae_workflows_created", 0),
            "mcae_tasks_executed": self._metrics.get("mcae_tasks_executed", 0),
            "mcae_adapter_available": self.mcae_adapter is not None
        }
        
        # Add MCAE adapter metrics if available
        if self.mcae_adapter:
            try:
                adapter_metrics = self.mcae_adapter.get_metrics()
                mcae_metrics["mcae_adapter_metrics"] = adapter_metrics
            except Exception as e:
                logger.error(f"Failed to get MCAE adapter metrics: {e}")
                mcae_metrics["mcae_adapter_error"] = str(e)
        
        return {**base_metrics, **mcae_metrics}
    
    async def health_check_mcae(self) -> Dict[str, Any]:
        """Perform health check on MCAE integration"""
        try:
            if not self.mcae_adapter:
                return {
                    "status": "unavailable",
                    "message": "MCAE adapter not initialized"
                }
            
            # Perform MCAE health check
            mcae_health = await self.mcae_adapter.health_check()
            
            return {
                "status": "healthy" if mcae_health.get("status") == "healthy" else "unhealthy",
                "mcae_health": mcae_health,
                "integration_active": True
            }
            
        except Exception as e:
            logger.error(f"MCAE health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "integration_active": False
            }