# forge1/backend/forge1/services/client_manager.py
"""
Client Management System

Handles client onboarding, configuration, and employee limit management
with complete tenant isolation and validation.

Requirements: 1.1, 1.3, 5.1, 5.2
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from decimal import Decimal

from forge1.core.database_config import DatabaseManager, get_database_manager
from forge1.core.tenancy import get_current_tenant, set_current_tenant
from forge1.models.employee_models import (
    Client, ClientInfo, ClientConfiguration, ClientStatus, ClientTier,
    SecurityLevel, EmployeeRequirements, Employee, EmployeeStatus,
    ClientNotFoundError, EmployeeLimitExceededError, InvalidConfigurationError,
    TenantIsolationError, generate_client_id, generate_employee_id,
    create_default_client_config, PersonalityConfig, ModelPreferences,
    CommunicationStyle, FormalityLevel, ExpertiseLevel, ResponseLength
)

logger = logging.getLogger(__name__)


class ClientManager:
    """
    Manages client onboarding, configuration, and employee lifecycle.
    
    Features:
    - Client onboarding with automatic configuration
    - Employee limit enforcement
    - Tenant isolation validation
    - Client status management
    - Employee creation from requirements
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager
        self._initialized = False
        self._client_cache = {}
        
    async def initialize(self):
        """Initialize the client manager"""
        if self._initialized:
            return
            
        if not self.db_manager:
            self.db_manager = await get_database_manager()
            
        self._initialized = True
        logger.info("Client Manager initialized")
    
    async def onboard_client(self, client_info: ClientInfo) -> Client:
        """
        Onboard a new client with initial configuration.
        
        Creates client record, initializes tenant namespace, and sets up
        default configurations based on tier and requirements.
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Generate unique client ID
            client_id = generate_client_id()
            
            # Create client configuration based on tier
            configuration = await self._create_client_configuration(client_info)
            
            # Create client entity
            client = Client(
                id=client_id,
                name=client_info.name,
                industry=client_info.industry,
                tier=client_info.tier,
                configuration=configuration,
                created_at=datetime.now(timezone.utc),
                status=ClientStatus.ACTIVE
            )
            
            # Store client in database
            await self._store_client(client)
            
            # Initialize client namespace and security
            await self._initialize_client_namespace(client_id)
            
            # Cache client for future access
            self._client_cache[client_id] = client
            
            logger.info(f"Successfully onboarded client {client_id}: {client_info.name}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to onboard client {client_info.name}: {e}")
            raise InvalidConfigurationError(f"Client onboarding failed: {e}")
    
    async def get_client(self, client_id: str) -> Optional[Client]:
        """
        Retrieve client by ID with tenant validation.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id)
        
        # Check cache first
        if client_id in self._client_cache:
            return self._client_cache[client_id]
        
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                row = await conn.fetchrow("""
                    SELECT 
                        id, name, industry, tier, configuration,
                        max_employees, allowed_models, security_level,
                        compliance_requirements, created_at, updated_at, status
                    FROM forge1_employees.clients
                    WHERE id = $1
                """, client_id)
            
            if not row:
                return None
            
            # Create client configuration
            configuration = ClientConfiguration(
                max_employees=row['max_employees'],
                allowed_models=row['allowed_models'],
                security_level=SecurityLevel(row['security_level']),
                compliance_requirements=row['compliance_requirements'],
                custom_settings=row['configuration'] or {}
            )
            
            # Create client entity
            client = Client(
                id=row['id'],
                name=row['name'],
                industry=row['industry'],
                tier=ClientTier(row['tier']),
                configuration=configuration,
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                status=ClientStatus(row['status'])
            )
            
            # Cache for future access
            self._client_cache[client_id] = client
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to retrieve client {client_id}: {e}")
            raise ClientNotFoundError(f"Failed to retrieve client: {e}")
    
    async def list_clients(
        self,
        status: Optional[ClientStatus] = None,
        tier: Optional[ClientTier] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Client]:
        """
        List clients with optional filtering.
        Note: This would typically be restricted to admin users.
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Build query conditions
            conditions = []
            params = []
            
            if status:
                conditions.append(f"status = ${len(params) + 1}")
                params.append(status.value)
            
            if tier:
                conditions.append(f"tier = ${len(params) + 1}")
                params.append(tier.value)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Add pagination
            params.extend([limit, offset])
            
            async with self.db_manager.postgres.acquire() as conn:
                rows = await conn.fetch(f"""
                    SELECT 
                        id, name, industry, tier, configuration,
                        max_employees, allowed_models, security_level,
                        compliance_requirements, created_at, updated_at, status
                    FROM forge1_employees.clients
                    {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${len(params) - 1} OFFSET ${len(params)}
                """, *params)
            
            clients = []
            for row in rows:
                configuration = ClientConfiguration(
                    max_employees=row['max_employees'],
                    allowed_models=row['allowed_models'],
                    security_level=SecurityLevel(row['security_level']),
                    compliance_requirements=row['compliance_requirements'],
                    custom_settings=row['configuration'] or {}
                )
                
                client = Client(
                    id=row['id'],
                    name=row['name'],
                    industry=row['industry'],
                    tier=ClientTier(row['tier']),
                    configuration=configuration,
                    created_at=row['created_at'],
                    updated_at=row['updated_at'],
                    status=ClientStatus(row['status'])
                )
                clients.append(client)
            
            return clients
            
        except Exception as e:
            logger.error(f"Failed to list clients: {e}")
            raise InvalidConfigurationError(f"Failed to list clients: {e}")
    
    async def update_client(
        self,
        client_id: str,
        updates: Dict[str, Any]
    ) -> Client:
        """
        Update client configuration.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id)
        
        try:
            # Get current client
            client = await self.get_client(client_id)
            if not client:
                raise ClientNotFoundError(f"Client {client_id} not found")
            
            # Prepare update fields
            update_fields = []
            update_params = []
            param_count = 0
            
            # Handle allowed updates
            allowed_fields = {
                'name': 'name',
                'industry': 'industry', 
                'tier': 'tier',
                'max_employees': 'max_employees',
                'allowed_models': 'allowed_models',
                'security_level': 'security_level',
                'compliance_requirements': 'compliance_requirements',
                'status': 'status'
            }
            
            for field, db_field in allowed_fields.items():
                if field in updates:
                    param_count += 1
                    update_fields.append(f"{db_field} = ${param_count}")
                    
                    # Handle enum values
                    value = updates[field]
                    if hasattr(value, 'value'):
                        value = value.value
                    
                    update_params.append(value)
            
            if not update_fields:
                return client  # No updates to apply
            
            # Add updated_at timestamp
            param_count += 1
            update_fields.append(f"updated_at = ${param_count}")
            update_params.append(datetime.now(timezone.utc))
            
            # Add client_id for WHERE clause
            param_count += 1
            update_params.append(client_id)
            
            # Execute update
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                await conn.execute(f"""
                    UPDATE forge1_employees.clients
                    SET {', '.join(update_fields)}
                    WHERE id = ${param_count}
                """, *update_params)
            
            # Clear cache and return updated client
            if client_id in self._client_cache:
                del self._client_cache[client_id]
            
            updated_client = await self.get_client(client_id)
            
            logger.info(f"Updated client {client_id}")
            return updated_client
            
        except Exception as e:
            logger.error(f"Failed to update client {client_id}: {e}")
            raise InvalidConfigurationError(f"Failed to update client: {e}")
    
    async def create_employee_for_client(
        self,
        client_id: str,
        requirements: EmployeeRequirements
    ) -> Employee:
        """
        Create an AI employee for a specific client based on requirements.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id)
        
        try:
            # Get and validate client
            client = await self.get_client(client_id)
            if not client or client.status != ClientStatus.ACTIVE:
                raise ClientNotFoundError(f"Client {client_id} not found or inactive")
            
            # Check employee limits
            current_employee_count = await self._get_client_employee_count(client_id)
            if current_employee_count >= client.configuration.max_employees:
                raise EmployeeLimitExceededError(
                    f"Client {client_id} has reached maximum employee limit of {client.configuration.max_employees}"
                )
            
            # Generate employee configuration from requirements
            employee_config = await self._generate_employee_from_requirements(
                client, requirements
            )
            
            # Create employee entity
            employee = Employee(
                id=employee_config['id'],
                client_id=client_id,
                name=employee_config['name'],
                role=employee_config['role'],
                personality=employee_config['personality'],
                model_preferences=employee_config['model_preferences'],
                tool_access=employee_config['tool_access'],
                knowledge_sources=employee_config['knowledge_sources'],
                created_at=datetime.now(timezone.utc),
                status=EmployeeStatus.ACTIVE
            )
            
            # Store employee in database
            await self._store_employee(employee)
            
            logger.info(f"Created employee {employee.id} for client {client_id}")
            
            return employee
            
        except (ClientNotFoundError, EmployeeLimitExceededError):
            raise
        except Exception as e:
            logger.error(f"Failed to create employee for client {client_id}: {e}")
            raise InvalidConfigurationError(f"Failed to create employee: {e}")
    
    async def get_client_employees(
        self,
        client_id: str,
        status: Optional[EmployeeStatus] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Employee]:
        """
        Get all employees for a client.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id)
        
        try:
            # Build query conditions
            conditions = ["client_id = $1"]
            params = [client_id]
            
            if status:
                conditions.append(f"status = ${len(params) + 1}")
                params.append(status.value)
            
            where_clause = " AND ".join(conditions)
            
            # Add pagination
            params.extend([limit, offset])
            
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                rows = await conn.fetch(f"""
                    SELECT 
                        id, client_id, name, role, personality, model_preferences,
                        tool_access, knowledge_sources, created_at, updated_at,
                        last_interaction_at, status, communication_style,
                        formality_level, expertise_level, response_length,
                        creativity_level, empathy_level, primary_model,
                        fallback_models, temperature, max_tokens, specialized_models
                    FROM forge1_employees.employees
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ${len(params) - 1} OFFSET ${len(params)}
                """, *params)
            
            employees = []
            for row in rows:
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
                employees.append(employee)
            
            return employees
            
        except Exception as e:
            logger.error(f"Failed to get employees for client {client_id}: {e}")
            raise ClientNotFoundError(f"Failed to get client employees: {e}")
    
    async def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """
        Get statistics for a client.
        """
        if not self._initialized:
            await self.initialize()
            
        # Validate tenant access
        await self._validate_tenant_access(client_id)
        
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                # Get employee stats
                employee_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_employees,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_employees,
                        COUNT(CASE WHEN last_interaction_at > NOW() - INTERVAL '24 hours' THEN 1 END) as recent_active,
                        MIN(created_at) as first_employee_created,
                        MAX(last_interaction_at) as last_activity
                    FROM forge1_employees.employees
                    WHERE client_id = $1
                """, client_id)
                
                # Get interaction stats
                interaction_stats = await conn.fetchrow("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT employee_id) as employees_with_interactions,
                        COUNT(DISTINCT session_id) as unique_sessions,
                        SUM(tokens_used) as total_tokens,
                        SUM(cost) as total_cost
                    FROM forge1_employees.employee_interactions
                    WHERE client_id = $1
                """, client_id)
                
                # Get employee role distribution
                role_distribution = await conn.fetch("""
                    SELECT role, COUNT(*) as count
                    FROM forge1_employees.employees
                    WHERE client_id = $1
                    GROUP BY role
                    ORDER BY count DESC
                """, client_id)
            
            return {
                "client_id": client_id,
                "total_employees": employee_stats['total_employees'] or 0,
                "active_employees": employee_stats['active_employees'] or 0,
                "recently_active_employees": employee_stats['recent_active'] or 0,
                "first_employee_created": employee_stats['first_employee_created'].isoformat() if employee_stats['first_employee_created'] else None,
                "last_activity": employee_stats['last_activity'].isoformat() if employee_stats['last_activity'] else None,
                "total_interactions": interaction_stats['total_interactions'] or 0,
                "employees_with_interactions": interaction_stats['employees_with_interactions'] or 0,
                "unique_sessions": interaction_stats['unique_sessions'] or 0,
                "total_tokens_used": int(interaction_stats['total_tokens'] or 0),
                "total_cost": float(interaction_stats['total_cost'] or Decimal('0.00')),
                "role_distribution": {row['role']: row['count'] for row in role_distribution}
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for client {client_id}: {e}")
            raise ClientNotFoundError(f"Failed to get client statistics: {e}")
    
    # Private helper methods
    
    async def _validate_tenant_access(self, client_id: str) -> None:
        """Validate that current tenant can access the client"""
        current_tenant = get_current_tenant()
        if current_tenant and current_tenant != client_id:
            raise TenantIsolationError(
                f"Tenant {current_tenant} cannot access client {client_id}"
            )
    
    async def _create_client_configuration(self, client_info: ClientInfo) -> ClientConfiguration:
        """Create client configuration based on tier and requirements"""
        
        # Base configuration
        config = ClientConfiguration(
            max_employees=client_info.max_employees,
            allowed_models=client_info.allowed_models,
            security_level=client_info.security_level,
            compliance_requirements=client_info.compliance_requirements
        )
        
        # Tier-specific adjustments
        if client_info.tier == ClientTier.ENTERPRISE:
            config.max_employees = max(config.max_employees, 100)
            config.allowed_models.extend(["gpt-4-turbo", "claude-3-opus"])
            config.security_level = SecurityLevel.HIGH
        elif client_info.tier == ClientTier.PROFESSIONAL:
            config.max_employees = max(config.max_employees, 50)
            config.allowed_models.extend(["claude-3-sonnet"])
        
        return config
    
    async def _store_client(self, client: Client) -> None:
        """Store client in database"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                await conn.execute("""
                    INSERT INTO forge1_employees.clients (
                        id, name, industry, tier, configuration,
                        max_employees, allowed_models, security_level,
                        compliance_requirements, created_at, updated_at, status
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                    client.id, client.name, client.industry, client.tier.value,
                    client.configuration.custom_settings,
                    client.configuration.max_employees,
                    client.configuration.allowed_models,
                    client.configuration.security_level.value,
                    client.configuration.compliance_requirements,
                    client.created_at, client.updated_at, client.status.value
                )
        except Exception as e:
            logger.error(f"Failed to store client {client.id}: {e}")
            raise
    
    async def _initialize_client_namespace(self, client_id: str) -> None:
        """Initialize client namespace and security settings"""
        try:
            # Set up tenant context for this client
            set_current_tenant(client_id)
            
            # Initialize any client-specific resources
            logger.debug(f"Initialized namespace for client {client_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize namespace for client {client_id}: {e}")
            raise
    
    async def _get_client_employee_count(self, client_id: str) -> int:
        """Get current employee count for client"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", client_id)
                
                count = await conn.fetchval("""
                    SELECT COUNT(*) FROM forge1_employees.employees
                    WHERE client_id = $1 AND status != 'archived'
                """, client_id)
                
                return count or 0
                
        except Exception as e:
            logger.error(f"Failed to get employee count for client {client_id}: {e}")
            return 0
    
    async def _generate_employee_from_requirements(
        self,
        client: Client,
        requirements: EmployeeRequirements
    ) -> Dict[str, Any]:
        """Generate employee configuration from client requirements"""
        
        # Generate unique employee ID
        employee_id = generate_employee_id(client.id)
        
        # Generate employee name based on role
        employee_name = await self._generate_employee_name(requirements.role)
        
        # Create personality configuration
        personality = PersonalityConfig(
            communication_style=requirements.communication_style,
            formality_level=requirements.formality_level,
            expertise_level=requirements.expertise_level,
            response_length=requirements.response_length,
            creativity_level=requirements.creativity_level,
            empathy_level=requirements.empathy_level,
            custom_traits=requirements.personality_traits
        )
        
        # Create model preferences
        model_preferences = ModelPreferences()
        if requirements.model_preferences:
            if 'primary_model' in requirements.model_preferences:
                model_preferences.primary_model = requirements.model_preferences['primary_model']
            if 'temperature' in requirements.model_preferences:
                model_preferences.temperature = requirements.model_preferences['temperature']
            if 'max_tokens' in requirements.model_preferences:
                model_preferences.max_tokens = requirements.model_preferences['max_tokens']
        
        # Ensure model is allowed for client
        if model_preferences.primary_model not in client.configuration.allowed_models:
            model_preferences.primary_model = client.configuration.allowed_models[0]
        
        # Configure tools based on requirements
        tool_access = requirements.tools_needed.copy()
        
        # Add industry-specific tools
        industry_tools = await self._get_industry_tools(requirements.industry)
        tool_access.extend(industry_tools)
        
        # Configure knowledge sources
        knowledge_sources = requirements.knowledge_domains.copy()
        
        return {
            'id': employee_id,
            'name': employee_name,
            'role': requirements.role,
            'personality': personality,
            'model_preferences': model_preferences,
            'tool_access': list(set(tool_access)),  # Remove duplicates
            'knowledge_sources': knowledge_sources
        }
    
    async def _generate_employee_name(self, role: str) -> str:
        """Generate a professional name for the employee based on role"""
        # In production, this could use more sophisticated name generation
        role_names = {
            'lawyer': ['Alex Legal', 'Jordan Counsel', 'Taylor Attorney'],
            'accountant': ['Morgan Finance', 'Casey Numbers', 'Riley Audit'],
            'analyst': ['Sam Analytics', 'Quinn Data', 'Avery Insights'],
            'consultant': ['Blake Advisory', 'Cameron Strategy', 'Drew Solutions'],
            'manager': ['Parker Leadership', 'Sage Operations', 'River Management']
        }
        
        # Simple role matching
        for key, names in role_names.items():
            if key.lower() in role.lower():
                return names[0]  # Return first name for consistency
        
        # Default naming pattern
        return f"AI {role.title()}"
    
    async def _get_industry_tools(self, industry: str) -> List[str]:
        """Get industry-specific tools"""
        industry_tools = {
            'legal': ['legal_research', 'document_review', 'case_analysis', 'contract_drafting'],
            'finance': ['financial_analysis', 'risk_assessment', 'portfolio_management', 'compliance_check'],
            'healthcare': ['medical_research', 'patient_data_analysis', 'diagnosis_support', 'treatment_planning'],
            'technology': ['code_review', 'system_analysis', 'security_audit', 'performance_optimization'],
            'marketing': ['market_research', 'campaign_analysis', 'content_creation', 'social_media_management'],
            'hr': ['resume_screening', 'interview_scheduling', 'performance_review', 'policy_management']
        }
        
        return industry_tools.get(industry.lower(), ['general_research', 'data_analysis'])
    
    async def _store_employee(self, employee: Employee) -> None:
        """Store employee in database"""
        try:
            async with self.db_manager.postgres.acquire() as conn:
                # Set RLS context
                await conn.execute("SELECT set_config('app.current_client_id', $1, true)", employee.client_id)
                
                await conn.execute("""
                    INSERT INTO forge1_employees.employees (
                        id, client_id, name, role, personality,
                        communication_style, formality_level, expertise_level,
                        response_length, creativity_level, empathy_level,
                        model_preferences, primary_model, fallback_models,
                        temperature, max_tokens, specialized_models,
                        tool_access, knowledge_sources, created_at,
                        updated_at, status
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20, $21, $22
                    )
                """,
                    employee.id, employee.client_id, employee.name, employee.role,
                    employee.personality.custom_traits,
                    employee.personality.communication_style.value,
                    employee.personality.formality_level.value,
                    employee.personality.expertise_level.value,
                    employee.personality.response_length.value,
                    employee.personality.creativity_level,
                    employee.personality.empathy_level,
                    employee.model_preferences.specialized_models,
                    employee.model_preferences.primary_model,
                    employee.model_preferences.fallback_models,
                    employee.model_preferences.temperature,
                    employee.model_preferences.max_tokens,
                    employee.model_preferences.specialized_models,
                    employee.tool_access, employee.knowledge_sources,
                    employee.created_at, employee.updated_at, employee.status.value
                )
                
        except Exception as e:
            logger.error(f"Failed to store employee {employee.id}: {e}")
            raise